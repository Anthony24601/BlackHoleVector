import faiss

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False)

        # FAISS index for approximate nearest neighbors using dot product
        self.key_db = faiss.IndexFlatIP(self.head_dim)
        self.value_store = {}

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.key_db.add(xk.cpu().numpy())
        for i in range(bsz * seqlen):
            self.value_store[i] = xv[i].cpu().numpy()

        distances, indices = self.key_db.search(xq.cpu().numpy(), k=10)
        top_k_keys = torch.tensor([self.key_db.reconstruct(i) for i in indices.flatten()]).to(xq.device)
        top_k_values = torch.tensor([self.value_store[i] for i in indices.flatten()]).to(xq.device)

        scores = torch.matmul(xq, top_k_keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, top_k_values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

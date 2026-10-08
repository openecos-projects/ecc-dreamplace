module tiny_path_parity (a, b, y_buf, y_inv);
  input a;
  input b;
  output y_buf;
  output y_inv;
  wire n_buf;
  wire n_inv;
  wire n_xor;

  BUF u_buf0 (.A(a), .Y(n_buf));
  INV u_inv0 (.A(n_buf), .Y(n_inv));
  XOR2 u_xor0 (.A(n_inv), .B(b), .Y(n_xor));
  BUF u_buf1 (.A(n_xor), .Y(y_buf));
  INV u_inv1 (.A(n_xor), .Y(y_inv));
endmodule

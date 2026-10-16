create_clock -name virtual_clk -period 50
set_input_delay 0 -clock virtual_clk [get_ports {a b}]
set_input_transition 1 [get_ports {a b}]
set_output_delay 45 -clock virtual_clk [get_ports {y_buf y_inv}]
set_load 0.001 [get_ports {y_buf y_inv}]

dirname = "/Users/miguelboland/Projects/uni/project_3/src/model_runner/benchmark/raw_imgs";
for i=1000:1000:17000
    wrap_script_simulated_structure_w_expansion(256, -1, i, string(i) + "_points_in.tif", string(i) + "_points_out.tif");
end
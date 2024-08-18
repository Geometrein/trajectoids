
$masterscale = 15.875;
$outergeorad = 1.3;
$numberofboxes = 120;

module cutter_cube(i) {
    import(str("cut_meshes/mesh_", i, ".stl"));
}

module geosphere(radius) {
    scale([radius, radius, radius]) import("unit_geosphere.stl");
}

scale([$masterscale, $masterscale, $masterscale]) difference() {
    geosphere(radius=$outergeorad);
    for (i = [0:$numberofboxes]) cutter_cube(i);
}

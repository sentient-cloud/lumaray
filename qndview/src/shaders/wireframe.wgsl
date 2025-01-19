struct Uniforms {
    mvpMatrix : mat4x4<f32>,
    cameraDir : vec4<f32>,
    hoverIndex : u32,
};
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct Output {
    @builtin(position) Position : vec4<f32>,
    @location(0) vNormal : vec4<f32>,
    @location(1) vColor : vec4<f32>,
    @location(2) uIndex : u32,
};

@vertex
fn vs_main(@location(0) pos: vec4<f32>,
           @location(1) normal: vec4<f32>,
           @location(2) color: vec4<f32>,
           @location(3) index: u32) -> Output {
    var output: Output;
    output.Position = uniforms.mvpMatrix * pos;
    output.vNormal = normal;
    output.vColor = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    output.uIndex = index;
    return output;
}

@fragment
fn fs_main(@location(0) vNormal: vec4<f32>,
           @location(1) vColor: vec4<f32>,
           @location(2) uIndex: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
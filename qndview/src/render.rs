use bytemuck::{Pod, Zeroable};
use cgmath::*;
use wgpu::{self, util::DeviceExt};

pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
);

pub fn create_view_projection(
    width: f32,
    height: f32,
    fov_y: f32,
    camera_pos: Point3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    near_plane: f32,
    far_plane: f32,
) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    let aspect = width / height;
    let view_matrix = Matrix4::look_at_rh(camera_pos, camera_pos + camera_dir, camera_up);
    let projection_matrix =
        OPENGL_TO_WGPU_MATRIX * perspective(Deg(fov_y), aspect, near_plane, far_plane);

    let view_project_mat = projection_matrix * view_matrix;

    (view_matrix, projection_matrix, view_project_mat)
}

pub fn create_model_transform(
    position: Point3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
) -> Matrix4<f32> {
    let translation = Matrix4::from_translation(position.to_vec());
    let rotation_x = Matrix4::from_angle_x(Rad(rotation.x));
    let rotation_y = Matrix4::from_angle_y(Rad(rotation.y));
    let rotation_z = Matrix4::from_angle_z(Rad(rotation.z));
    let scale = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);

    translation * rotation_x * rotation_y * rotation_z * scale
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PaintedFaceVertex {
    pub position: [f32; 4],
    pub normals: [f32; 4],
    pub color: [f32; 4],
    pub index: u32,
}

impl PaintedFaceVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Uint32
    ];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderStateParams {
    pub width: f32,
    pub height: f32,
    pub fov_y: f32,
    pub camera_pos: Point3<f32>,
    pub camera_dir: Vector3<f32>,
    pub camera_up: Vector3<f32>,
    pub near_plane: f32,
    pub far_plane: f32,
}

pub struct RenderState {
    pub facet_pipeline: wgpu::RenderPipeline,
    pub wireframe_pipeline: wgpu::RenderPipeline,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_lens: Vec<u32>,
    // pub line_buffer: wgpu::Buffer,
    // pub line_buffer_len: u32,
    pub vertices: Vec<PaintedFaceVertex>,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub depth_buffer: ggez::graphics::ScreenImage,
    pub model_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub camera_dir: Vector3<f32>,
    pub hovered_triangle: Option<u32>,
    pub projection_matrix: Matrix4<f32>,
    pub skip_frame: usize,
}

impl RenderState {
    pub fn new(
        ctx: &mut ggez::Context,
        params: RenderStateParams,
        model: Vec<PaintedFaceVertex>,
    ) -> RenderState {
        let device = &ctx.gfx.wgpu().device;
        let surface_format = ctx.gfx.surface_format();

        let facet_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("facet_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/facet.wgsl").into()),
        });

        let wireframe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wireframe_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wireframe.wgsl").into()),
        });

        let model_matrix = create_model_transform(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let (view_matrix, projection_matrix, view_projection_matrix) = create_view_projection(
            params.width,
            params.height,
            params.fov_y,
            params.camera_pos,
            params.camera_dir,
            params.camera_up,
            params.near_plane,
            params.far_plane,
        );

        let mvp_matrix = view_projection_matrix * model_matrix;
        let mvp_ref: &[f32; 16] = mvp_matrix.as_ref();

        let mut uniform = [0.0; 24];
        uniform[0..16].copy_from_slice(mvp_ref);
        uniform[16] = params.camera_dir.x;
        uniform[17] = params.camera_dir.y;
        uniform[18] = params.camera_dir.z;
        uniform[19] = 0.0;
        uniform[20] = bytemuck::cast(!(0u32));

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: bytemuck::cast_slice(&uniform[..]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let facet_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("facet_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &facet_shader,
                entry_point: "vs_main",
                buffers: &[PaintedFaceVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &facet_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &wireframe_shader,
                entry_point: "vs_main",
                buffers: &[PaintedFaceVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &wireframe_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_lens = Vec::new();

        for verts in model.chunks(3 * 1024 * 128) {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_lens.push(verts.len() as u32);
        }

        let depth_buffer = ggez::graphics::ScreenImage::new(
            ctx,
            ggez::graphics::ImageFormat::Depth32Float,
            1.0,
            1.0,
            1,
        );

        RenderState {
            facet_pipeline,
            wireframe_pipeline,
            vertex_buffers,
            vertex_buffer_lens,
            // line_buffer,
            // line_buffer_len: lines.len() as u32,
            vertices: model.clone(),
            uniform_buffer,
            uniform_bind_group,
            depth_buffer,
            model_matrix,
            view_matrix,
            camera_dir: params.camera_dir,
            hovered_triangle: None,
            projection_matrix,
            skip_frame: 1,
        }
    }

    pub fn resize(
        &mut self,
        ctx: &ggez::Context,
        fov_y: f32,
        width: f32,
        height: f32,
        near_plane: f32,
        far_plane: f32,
    ) {
        let aspect = width / height;
        let projection_matrix =
            OPENGL_TO_WGPU_MATRIX * perspective(Deg(fov_y), aspect, near_plane, far_plane);

        self.projection_matrix = projection_matrix;
        self.update_mvp(ctx);

        self.skip_frame = 1;
    }

    pub fn set_camera_dir(&mut self, camera_dir: Vector3<f32>) {
        self.camera_dir = camera_dir;
    }

    pub fn set_hovered_triangle(&mut self, triangle: Option<u32>) {
        self.hovered_triangle = triangle;
    }

    pub fn update_mvp(&mut self, ctx: &ggez::Context) {
        let mvp_matrix = self.projection_matrix * self.view_matrix * self.model_matrix;
        let mvp_ref: &[f32; 16] = mvp_matrix.as_ref();

        let mut uniform = [0.0; 24];
        uniform[0..16].copy_from_slice(mvp_ref);
        uniform[16] = self.camera_dir.x;
        uniform[17] = self.camera_dir.y;
        uniform[18] = self.camera_dir.z;
        uniform[19] = 0.0;
        uniform[20] = bytemuck::cast(self.hovered_triangle.unwrap_or(!(0u32)));

        let queue = &ctx.gfx.wgpu().queue;

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform[..]));
    }

    pub fn set_view_matrix(&mut self, ctx: &ggez::Context, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.update_mvp(ctx);
    }

    #[allow(dead_code)]
    pub fn set_model_matrix(&mut self, ctx: &ggez::Context, model_matrix: Matrix4<f32>) {
        self.model_matrix = model_matrix;
        self.update_mvp(ctx);
    }

    #[allow(dead_code)]
    pub fn set_projection_matrix(&mut self, ctx: &ggez::Context, projection_matrix: Matrix4<f32>) {
        self.projection_matrix = projection_matrix;
        self.update_mvp(ctx);
    }

    pub fn set_vertex_buffer(&mut self, ctx: &ggez::Context, model: Vec<PaintedFaceVertex>) {
        let device = &ctx.gfx.wgpu().device;

        self.vertices = model.clone();

        for buffer in &self.vertex_buffers {
            buffer.destroy();
        }

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_lens = Vec::new();

        for (i, verts) in model.chunks(3 * 1024 * 128).enumerate() {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("vertex_buffer_{}", i).as_str()),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_lens.push(verts.len() as u32);
        }

        self.vertex_buffers = vertex_buffers;
        self.vertex_buffer_lens = vertex_buffer_lens;

        // let mut lines = Vec::new();
        // for i in (0..model.len()).step_by(3) {
        //     lines.push(model[i]);
        //     lines.push(model[i + 1]);
        //     lines.push(model[i + 2]);
        //     lines.push(model[i]);
        // }

        // self.line_buffer.destroy();
        // self.line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("line_buffer"),
        //     contents: bytemuck::cast_slice(&lines),
        //     usage: wgpu::BufferUsages::VERTEX,
        // });

        // self.line_buffer_len = lines.len() as u32;
    }

    pub fn draw(&mut self, ctx: &mut ggez::Context) {
        let depth = self.depth_buffer.image(ctx);
        let frame = ctx.gfx.frame().clone();

        if self.skip_frame == 0 {
            if frame.width() != depth.width() || frame.height() != depth.height() {
                self.skip_frame += 1;
            } else {
                let commands = ctx.gfx.commands().unwrap();

                let mut render_pass = commands.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.wgpu().1,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth.wgpu().1,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: false,
                        }),
                        stencil_ops: None,
                    }),
                });

                for (i, buffer) in self.vertex_buffers.iter().enumerate() {
                    render_pass.set_pipeline(&self.facet_pipeline);
                    render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    render_pass.draw(0..self.vertex_buffer_lens[i], 0..1);
                }

                // render_pass.set_pipeline(&self.wireframe_pipeline);
                // render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                // render_pass.set_vertex_buffer(0, self.line_buffer.slice(..));
                // render_pass.draw(0..self.line_buffer_len, 0..1);
            }
        } else {
            self.skip_frame -= 1;
        }
    }
}

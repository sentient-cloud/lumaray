use cgmath::*;
use ggez::event::{self, EventHandler};
use ggez::graphics::{self, Drawable};
use ggez::{Context, ContextBuilder, GameResult};
use rand::Rng;
use render::{PaintedFaceVertex, RenderState, RenderStateParams};

mod demo_models;
mod render;
mod stl;

use std::env;

fn main() {
    println!("qndview!!~~");

    let (mut ctx, event_loop) = ContextBuilder::new("qndview", "me uwu")
        .window_setup(ggez::conf::WindowSetup::default().title("qndview"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(1280.0, 960.0))
        .build()
        .expect("Context build failed");

    let mut qndview = QndView::new(&mut ctx);

    let args = env::args().collect::<Vec<String>>();

    if args.len() > 1 {
        let stl_path = &args[1];
        let mut bufreader = std::io::BufReader::new(std::fs::File::open(stl_path).unwrap());
        let stl = stl::STL::new_from_bufreader(&mut bufreader).unwrap();

        qndview.model_name = stl_path.clone();

        fn u16_to_rgb1(color: u16) -> [f32; 4] {
            let r = ((color >> 11) & 0x1F) as f32 / 31.0; // 5 bits
            let g = ((color >> 5) & 0x3F) as f32 / 63.0; // 6 bits
            let b = (color & 0x1F) as f32 / 31.0; // 5 bits
            [r, g, b, 1.0]
        }

        let vertices = stl
            .vertices
            .iter()
            .enumerate()
            .map(|(i, vert)| {
                let face_color = if vert.attribute == 0 {
                    random_rgb1()
                } else {
                    u16_to_rgb1(vert.attribute)
                };

                [
                    PaintedFaceVertex {
                        position: [
                            vert.vertices[0].x,
                            vert.vertices[0].z,
                            vert.vertices[0].y,
                            1.0,
                        ],
                        normals: [vert.normal.x, vert.normal.z, vert.normal.y, 0.0],
                        color: face_color,
                        index: (i as u32) / 3,
                    },
                    PaintedFaceVertex {
                        position: [
                            vert.vertices[1].x,
                            vert.vertices[1].z,
                            vert.vertices[1].y,
                            1.0,
                        ],
                        normals: [vert.normal.x, vert.normal.z, vert.normal.y, 0.0],
                        color: face_color,
                        index: (i as u32) / 3,
                    },
                    PaintedFaceVertex {
                        position: [
                            vert.vertices[2].x,
                            vert.vertices[2].z,
                            vert.vertices[2].y,
                            1.0,
                        ],
                        normals: [vert.normal.x, vert.normal.z, vert.normal.y, 0.0],
                        color: face_color,
                        index: (i as u32) / 3,
                    },
                ]
            })
            .flatten()
            .collect::<Vec<_>>();

        let (min, max) = {
            let mut min = Point3::<f32>::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
            let mut max =
                Point3::<f32>::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

            for vert in vertices.iter() {
                min.x = min.x.min(vert.position[0]);
                min.y = min.y.min(vert.position[1]);
                min.z = min.z.min(vert.position[2]);

                max.x = max.x.max(vert.position[0]);
                max.y = max.y.max(vert.position[1]);
                max.z = max.z.max(vert.position[2]);
            }

            (min, max)
        };

        let size = (max - min).magnitude();

        qndview.camera_distance = size;
        qndview.camera_distance_limit = size * 3.0;
        qndview.camera_pivot = (min + Vector3::<f32>::new(max.x, max.y, max.z)) / 2.0;
        qndview.camera_pos = qndview.camera_pivot - qndview.camera_dir * qndview.camera_distance;

        let view_matrix =
            Matrix4::look_at_rh(qndview.camera_pos, qndview.camera_pivot, qndview.camera_up);

        qndview.renderer.set_vertex_buffer(&ctx, vertices);
        qndview.renderer.set_view_matrix(&ctx, view_matrix);
    }

    event::run(ctx, event_loop, qndview);
}

fn random_rgb1() -> [f32; 4] {
    let mut rng = rand::thread_rng();
    [
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        1.0,
    ]
}

struct QndView {
    window_width: f32,
    window_height: f32,
    window_has_focus: bool,

    last_time: std::time::Instant,
    model_name: String,

    camera_fov_y: f32,
    camera_pos: Point3<f32>,
    camera_dir: Vector3<f32>,
    camera_up: Vector3<f32>,
    camera_pivot: Point3<f32>,
    camera_distance: f32,
    camera_distance_limit: f32,
    camera_pitch: f32,
    camera_yaw: f32,

    projection_near_plane: f32,
    projection_far_plane: f32,

    mouse_left_down: bool,
    mouse_middle_down: bool,
    mouse_right_down: bool,
    mouse_pos: Point2<f32>,
    mouse_ray_dir: Vector3<f32>,
    mouse_ray_hover: Option<usize>,

    renderer: RenderState,
}

impl QndView {
    pub fn new(ctx: &mut Context) -> QndView {
        ctx.gfx
            .set_resizable(true)
            .expect("Failed to set resizable");

        let fov_y = 60.0;

        let camera_pitch: f32 = -15.0;
        let camera_yaw: f32 = 225.0;

        let camera_pivot = Point3::<f32>::new(0.0, 0.0, 0.0);
        let camera_dir = Vector3::<f32>::new(
            camera_pitch.to_radians().cos() * camera_yaw.to_radians().cos(),
            camera_pitch.to_radians().sin(),
            camera_pitch.to_radians().cos() * camera_yaw.to_radians().sin(),
        );

        let camera_distance = 7.0;
        let camera_distance_limit = 21.0;

        let camera_pos = camera_pivot - camera_dir * camera_distance;
        let camera_up = Vector3::<f32>::new(0.0, 1.0, 0.0);

        let near_plane = 0.1;
        let far_plane = 10000.0;

        let render_params = RenderStateParams {
            width: ctx.gfx.size().0,
            height: ctx.gfx.size().1,
            fov_y,
            camera_pos,
            camera_dir,
            camera_up,
            near_plane,
            far_plane,
        };

        let model = demo_models::create_vertices(
            &demo_models::cube::positions(),
            &demo_models::cube::colors(),
        );

        QndView {
            window_width: ctx.gfx.size().0,
            window_height: ctx.gfx.size().0,
            window_has_focus: true,
            last_time: std::time::Instant::now(),
            model_name: "cube (demo)".to_string(),
            camera_fov_y: fov_y,
            camera_pos,
            camera_dir,
            camera_up,
            camera_pivot,
            camera_distance,
            camera_distance_limit,
            camera_pitch,
            camera_yaw,
            projection_near_plane: near_plane,
            projection_far_plane: far_plane,
            mouse_left_down: false,
            mouse_middle_down: false,
            mouse_right_down: false,
            mouse_pos: Point2::new(0.0, 0.0),
            mouse_ray_dir: Vector3::new(0.0, 0.0, 0.0),
            mouse_ray_hover: None,
            renderer: RenderState::new(ctx, render_params, model),
        }
    }
}

fn intersect_ray_triangle(
    ray_origin: Point3<f32>,
    ray_dir: Vector3<f32>,
    triangle: [Point3<f32>; 3],
) -> Option<f32> {
    let edge1 = triangle[1] - triangle[0];
    let edge2 = triangle[2] - triangle[0];
    let pvec = ray_dir.cross(edge2);

    let det = edge1.dot(pvec);

    if det.abs() < 0.0001 {
        return None;
    }

    let inv_det = 1.0 / det;

    let tvec = ray_origin - triangle[0];
    let qvec = tvec.cross(edge1);

    let u = tvec.dot(pvec) * inv_det;
    let v = ray_dir.dot(qvec) * inv_det;

    if u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = edge2.dot(qvec) * inv_det;

    if t < 0.0 {
        return None;
    }

    Some(t)
}

impl EventHandler for QndView {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn focus_event(&mut self, _ctx: &mut Context, gained: bool) -> Result<(), ggez::GameError> {
        self.window_has_focus = gained;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        button: event::MouseButton,
        x: f32,
        y: f32,
    ) -> Result<(), ggez::GameError> {
        if button == event::MouseButton::Left {
            self.mouse_left_down = true;
        }

        if button == event::MouseButton::Middle {
            self.mouse_middle_down = true;
        }

        if button == event::MouseButton::Right {
            self.mouse_right_down = true;
        }

        self.mouse_pos = Point2::new(x, y);
        Ok(())
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        button: event::MouseButton,
        x: f32,
        y: f32,
    ) -> Result<(), ggez::GameError> {
        if button == event::MouseButton::Left {
            self.mouse_left_down = false;
        }

        if button == event::MouseButton::Middle {
            self.mouse_middle_down = false;
        }

        if button == event::MouseButton::Right {
            self.mouse_right_down = false;
        }

        self.mouse_pos = Point2::new(x, y);
        Ok(())
    }

    fn mouse_motion_event(
        &mut self,
        ctx: &mut Context,
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
    ) -> Result<(), ggez::GameError> {
        self.mouse_pos = Point2::new(x, y);

        // let camera_right = self.camera_dir.cross(self.camera_up).normalize();
        // let camera_up = camera_right.cross(self.camera_dir).normalize();

        // let aspect = self.window_width / self.window_height;
        // let xi = (x / self.window_width - 0.5) * 2.0 * aspect;
        // let yi = (y / self.window_height - 0.5) * 2.0;

        // let mouse_ray_dir = self.camera_dir
        //     + camera_right * xi * self.camera_fov_y.to_radians().tan()
        //     - camera_up * yi * self.camera_fov_y.to_radians().tan();

        // self.mouse_ray_dir = mouse_ray_dir.normalize();

        // let mut hit_index = usize::MAX;
        // let mut hit_t = f32::INFINITY;

        // for face in self.renderer.vertices.chunks(3) {
        //     let triangle = [
        //         Point3::new(
        //             face[0].position[0],
        //             face[0].position[1],
        //             face[0].position[2],
        //         ),
        //         Point3::new(
        //             face[1].position[0],
        //             face[1].position[1],
        //             face[1].position[2],
        //         ),
        //         Point3::new(
        //             face[2].position[0],
        //             face[2].position[1],
        //             face[2].position[2],
        //         ),
        //     ];

        //     if let Some(t) = intersect_ray_triangle(self.camera_pos, self.mouse_ray_dir, triangle) {
        //         if t < hit_t {
        //             hit_t = t;
        //             hit_index = face[0].index as usize;
        //         }
        //     }
        // }

        // if hit_index != usize::MAX {
        //     self.mouse_ray_hover = Some(hit_index);
        //     // println!("hit_index: {}", hit_index);
        // } else {
        //     self.mouse_ray_hover = None;
        // }

        // self.renderer
        //     .set_hovered_triangle(if hit_index != usize::MAX {
        //         Some(hit_index as u32)
        //     } else {
        //         None
        //     });

        if self.mouse_left_down {
            self.camera_yaw += dx / 3.0;
            self.camera_pitch -= dy / 3.0;
            self.camera_pitch = self.camera_pitch.clamp(-85.0, 85.0);

            self.camera_dir = Vector3::<f32>::new(
                self.camera_pitch.to_radians().cos() * self.camera_yaw.to_radians().cos(),
                self.camera_pitch.to_radians().sin(),
                self.camera_pitch.to_radians().cos() * self.camera_yaw.to_radians().sin(),
            );

            self.camera_pos = self.camera_pivot - self.camera_dir * self.camera_distance;

            let view_matrix =
                Matrix4::look_at_rh(self.camera_pos, self.camera_pivot, self.camera_up);

            self.renderer.set_camera_dir(self.camera_dir);
            self.renderer.set_view_matrix(ctx, view_matrix);
        } else if self.mouse_right_down {
            let camera_right = self.camera_dir.cross(self.camera_up).normalize();
            let camera_up = camera_right.cross(self.camera_dir).normalize();

            self.camera_pivot -= self.camera_distance * camera_right * dx / 1000.0;
            self.camera_pivot += self.camera_distance * camera_up * dy / 1000.0;

            self.camera_pos = self.camera_pivot - self.camera_dir * self.camera_distance;

            let view_matrix =
                Matrix4::look_at_rh(self.camera_pos, self.camera_pivot, self.camera_up);

            self.renderer.set_view_matrix(ctx, view_matrix);
        } else {
            self.renderer.update_mvp(&ctx);
        }

        Ok(())
    }

    fn mouse_wheel_event(
        &mut self,
        ctx: &mut Context,
        _x: f32,
        y: f32,
    ) -> Result<(), ggez::GameError> {
        self.camera_distance *= 1.0 - y / 10.0;
        self.camera_distance = self.camera_distance.clamp(0.1, self.camera_distance_limit);

        self.camera_pos = self.camera_pivot - self.camera_dir * self.camera_distance;
        let view_matrix = Matrix4::look_at_rh(self.camera_pos, self.camera_pivot, self.camera_up);

        self.renderer.set_view_matrix(ctx, view_matrix);

        Ok(())
    }

    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> Result<(), ggez::GameError> {
        match input {
            ggez::input::keyboard::KeyInput {
                scancode: _,
                keycode: Some(keycode),
                mods: _,
                ..
            } => {
                if keycode == ggez::input::keyboard::KeyCode::Escape {
                    ctx.quit_requested = true;
                }
            }

            _ => (),
        }
        Ok(())
    }

    fn resize_event(
        &mut self,
        ctx: &mut Context,
        width: f32,
        height: f32,
    ) -> Result<(), ggez::GameError> {
        self.window_width = width;
        self.window_height = height;
        self.renderer.resize(
            ctx,
            self.camera_fov_y,
            self.window_width,
            self.window_height,
            self.projection_near_plane,
            self.projection_far_plane,
        );
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let now = std::time::Instant::now();
        let delta_time = (now - self.last_time).as_secs_f32();
        self.last_time = now;

        self.renderer.draw(ctx);
        let mut canvas = graphics::Canvas::from_frame(ctx, None);

        let mut y = 5.0;

        let text = ggez::graphics::Text::new(format!("FPS: {:.2}", 1.0 / delta_time));
        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 20.0;
        let text = ggez::graphics::Text::new(format!("Model: {}", self.model_name));
        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 20.0;
        let text =
            ggez::graphics::Text::new(format!("Triangles: {}", self.renderer.vertices.len() / 3));
        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 30.0;
        let text = ggez::graphics::Text::new(format!(
            "Camera pos: [{:.2}, {:.2}, {:.2}]",
            self.camera_pos.x, self.camera_pos.y, self.camera_pos.z,
        ));

        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 20.0;
        let text = ggez::graphics::Text::new(format!(
            "Camera dir: [{:.2}, {:.2}, {:.2}]",
            self.camera_dir.x, self.camera_dir.y, self.camera_dir.z,
        ));

        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 20.0;
        let text = ggez::graphics::Text::new(format!(
            "Mouse ray: [{:.2}, {:.2}, {:.2}]",
            self.mouse_ray_dir.x, self.mouse_ray_dir.y, self.mouse_ray_dir.z,
        ));

        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        y += 20.0;
        let text = ggez::graphics::Text::new(format!("Mouse hover: {:?}", self.mouse_ray_hover));

        text.draw(
            &mut canvas,
            ggez::graphics::DrawParam {
                color: ggez::graphics::Color::WHITE,
                transform: ggez::graphics::Transform::Values {
                    offset: ggez::mint::Point2 { x: 0.0, y: 0.0 },
                    rotation: 0.0,
                    dest: ggez::mint::Point2 { x: 5.0, y },
                    scale: ggez::mint::Vector2 { x: 1.0, y: 1.0 },
                },
                ..Default::default()
            },
        );

        canvas.finish(ctx)
    }
}

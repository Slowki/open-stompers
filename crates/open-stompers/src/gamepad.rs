use gilrs::{Axis, Button, EventType, GamepadId, Gilrs, GilrsBuilder};
use open_stompers_hardware::GamepadCommand;

const SDL_GAMECONTROLLER_MAPPINGS: &str = include_str!("../configuration/beitong_a2p3a.txt");

pub(super) struct GamepadInput {
    gilrs: Gilrs,
    active_gamepad: Option<GamepadId>,
}

impl GamepadInput {
    pub(super) fn new() -> Result<Self, Box<gilrs::Error>> {
        Ok(Self {
            gilrs: GilrsBuilder::new()
                .add_included_mappings(true)
                .add_mappings(SDL_GAMECONTROLLER_MAPPINGS)
                .build()
                .map_err(Box::new)?,
            active_gamepad: None,
        })
    }

    pub(super) fn command(&mut self) -> GamepadCommand {
        while let Some(event) = self.gilrs.next_event() {
            match event.event {
                EventType::Connected if self.active_gamepad.is_none() => {
                    self.active_gamepad = Some(event.id);
                    tracing::info!(gamepad_id = ?event.id, "Gamepad connected");
                }
                EventType::Disconnected if self.active_gamepad == Some(event.id) => {
                    self.active_gamepad = None;
                    tracing::info!(gamepad_id = ?event.id, "Gamepad disconnected");
                }
                _ => {}
            }
        }

        self.active_gamepad = self
            .active_gamepad
            .filter(|gamepad_id| self.gilrs.connected_gamepad(*gamepad_id).is_some())
            .or_else(|| {
                self.gilrs
                    .gamepads()
                    .next()
                    .map(|(gamepad_id, _)| gamepad_id)
            });

        let Some(gamepad_id) = self.active_gamepad else {
            return GamepadCommand::default();
        };
        let gamepad = self.gilrs.gamepad(gamepad_id);
        GamepadCommand {
            forward_axis: gamepad.value(Axis::LeftStickY).clamp(-1.0, 1.0),
            lateral_axis: -gamepad.value(Axis::LeftStickX).clamp(-1.0, 1.0),
            yaw_axis: -gamepad.value(Axis::RightStickX).clamp(-1.0, 1.0),
            mode_toggle: gamepad.is_pressed(Button::RightThumb),
        }
    }
}

use std::{
    io::{self, Write},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use gilrs::ev::Code;
use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs, GilrsBuilder, Mapping};

const SDL_GAMECONTROLLER_MAPPINGS: &str = include_str!("../../configuration/beitong_a2p3a.txt");

#[derive(Debug, Parser)]
#[command(about = "Inspect gamepad inputs and create a Gilrs/SDL mapping")]
struct Arguments {
    /// Run the interactive mapping wizard and print an SDL_GAMECONTROLLERCONFIG line.
    #[arg(long)]
    create_mapping: bool,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let mut gilrs = GilrsBuilder::new()
        .add_included_mappings(true)
        .add_mappings(SDL_GAMECONTROLLER_MAPPINGS)
        .build()
        .map_err(|error| anyhow!("Failed to initialize Gilrs: {error}"))?;
    let gamepad_id = wait_for_gamepad(&mut gilrs)?;

    print_gamepad_details(&gilrs, gamepad_id);

    if arguments.create_mapping {
        create_mapping(&mut gilrs, gamepad_id)
    } else {
        monitor_inputs(&mut gilrs, gamepad_id)
    }
}

fn wait_for_gamepad(gilrs: &mut Gilrs) -> Result<GamepadId> {
    println!("Waiting for a gamepad...");

    loop {
        if let Some((gamepad_id, _)) = gilrs.gamepads().next() {
            return Ok(gamepad_id);
        }

        if let Some(event) = gilrs.next_event_blocking(Some(Duration::from_secs(1)))
            && matches!(event.event, EventType::Connected)
        {
            return Ok(event.id);
        }
    }
}

fn print_gamepad_details(gilrs: &Gilrs, gamepad_id: GamepadId) {
    let gamepad = gilrs.gamepad(gamepad_id);

    println!("Gamepad: {} (id {})", gamepad.name(), gamepad_id);
    println!("GUID: {}", format_guid(gamepad.uuid()));
    println!("Mapping source: {:?}", gamepad.mapping_source());
    println!("Vendor ID: {:?}", gamepad.vendor_id());
    println!("Product ID: {:?}", gamepad.product_id());
}

fn format_guid(bytes: [u8; 16]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn monitor_inputs(gilrs: &mut Gilrs, gamepad_id: GamepadId) -> Result<()> {
    println!("Move one stick or press one button at a time. Press Ctrl-C to quit.");

    loop {
        let Some(event) = gilrs.next_event_blocking(None) else {
            continue;
        };
        if event.id != gamepad_id {
            continue;
        }

        print_event(event);
    }
}

fn print_event(event: Event) {
    match event.event {
        EventType::AxisChanged(axis, value, code) => println!(
            "axis   mapped={axis:?} value={value:+.3} native_code=0x{:x}",
            code.into_u32()
        ),
        EventType::ButtonChanged(button, value, code) => println!(
            "button mapped={button:?} value={value:.3} native_code=0x{:x}",
            code.into_u32()
        ),
        EventType::ButtonPressed(button, code) => println!(
            "button pressed mapped={button:?} native_code=0x{:x}",
            code.into_u32()
        ),
        EventType::ButtonReleased(button, code) => println!(
            "button released mapped={button:?} native_code=0x{:x}",
            code.into_u32()
        ),
        other => println!("{other:?}"),
    }
}

fn create_mapping(gilrs: &mut Gilrs, gamepad_id: GamepadId) -> Result<()> {
    println!("\nEnter a target and move or press that control when prompted.");
    println!("Axis targets: leftx, lefty, leftz, rightx, righty, rightz");
    println!("Button targets: a, b, x, y, leftshoulder, rightshoulder, back, start,");
    println!("                guide, leftstick, rightstick, dpup, dpdown, dpleft, dpright");
    println!("Use done to finish. For your yaw stick, choose axis rightx.");

    let mut mapping = Mapping::new();

    loop {
        let command = read_line("map> ")?;
        let mut parts = command.split_whitespace();
        let Some(kind) = parts.next() else {
            continue;
        };

        if kind == "done" {
            break;
        }

        let Some(target_name) = parts.next() else {
            println!("Enter axis <name>, button <name>, or done.");
            continue;
        };

        if parts.next().is_some() {
            println!("Enter one target name, such as axis leftz.");
            continue;
        }

        match (
            kind,
            parse_axis_target(target_name),
            parse_button_target(target_name),
        ) {
            ("axis", Some(axis), _) => {
                let code = capture_axis(gilrs, gamepad_id, target_name)?;
                mapping.insert_axis(code, axis);
                println!(
                    "Mapped axis {target_name} from native code 0x{:x}.",
                    code.into_u32()
                );
            }
            ("button", _, Some(button)) => {
                let code = capture_button(gilrs, gamepad_id, target_name)?;
                mapping.insert_btn(code, button);
                println!(
                    "Mapped button {target_name} from native code 0x{:x}.",
                    code.into_u32()
                );
            }
            ("axis", _, _) => println!("Unknown axis target {target_name}."),
            ("button", _, _) => println!("Unknown button target {target_name}."),
            _ => println!("Use axis, button, or done as the command."),
        }
    }

    if mapping_is_empty(&mapping) {
        bail!("No controls were mapped");
    }

    let mapping_line = gilrs
        .set_mapping(usize::from(gamepad_id), &mapping, "Beitong Custom")
        .context("Gilrs could not generate a mapping from the captured controls")?;

    println!("\nGenerated mapping line:");
    println!("{mapping_line}");
    println!("\nCopy that line into SDL_GAMECONTROLLERCONFIG before starting the robot.");

    Ok(())
}

fn mapping_is_empty(mapping: &Mapping) -> bool {
    [
        Axis::LeftStickX,
        Axis::LeftStickY,
        Axis::LeftZ,
        Axis::RightStickX,
        Axis::RightStickY,
        Axis::RightZ,
    ]
    .iter()
    .all(|axis| mapping.axis(*axis).is_none())
        && [
            Button::South,
            Button::East,
            Button::West,
            Button::North,
            Button::LeftTrigger,
            Button::RightTrigger,
            Button::Select,
            Button::Start,
            Button::Mode,
            Button::LeftThumb,
            Button::RightThumb,
            Button::DPadUp,
            Button::DPadDown,
            Button::DPadLeft,
            Button::DPadRight,
        ]
        .iter()
        .all(|button| mapping.button(*button).is_none())
}

fn capture_axis(gilrs: &mut Gilrs, gamepad_id: GamepadId, target: &str) -> Result<Code> {
    println!("Move only the {target} axis through its full range within 30 seconds.");

    loop {
        let event = gilrs
            .next_event_blocking(Some(Duration::from_secs(30)))
            .with_context(|| format!("Timed out waiting for axis {target}"))?;

        if event.id == gamepad_id
            && let EventType::AxisChanged(_, value, code) = event.event
            && value.abs() >= 0.6
        {
            return Ok(code);
        }
    }
}

fn capture_button(gilrs: &mut Gilrs, gamepad_id: GamepadId, target: &str) -> Result<Code> {
    println!("Press only the {target} button within 30 seconds.");

    loop {
        let event = gilrs
            .next_event_blocking(Some(Duration::from_secs(30)))
            .with_context(|| format!("Timed out waiting for button {target}"))?;

        if event.id != gamepad_id {
            continue;
        }

        match event.event {
            EventType::ButtonPressed(_, code) => return Ok(code),
            EventType::ButtonChanged(_, value, code) if value >= 0.75 => return Ok(code),
            _ => {}
        }
    }
}

fn parse_axis_target(target: &str) -> Option<Axis> {
    match target {
        "leftx" => Some(Axis::LeftStickX),
        "lefty" => Some(Axis::LeftStickY),
        "leftz" => Some(Axis::LeftZ),
        "rightx" => Some(Axis::RightStickX),
        "righty" => Some(Axis::RightStickY),
        "rightz" => Some(Axis::RightZ),
        _ => None,
    }
}

fn parse_button_target(target: &str) -> Option<Button> {
    match target {
        "a" => Some(Button::South),
        "b" => Some(Button::East),
        "x" => Some(Button::West),
        "y" => Some(Button::North),
        "leftshoulder" => Some(Button::LeftTrigger),
        "rightshoulder" => Some(Button::RightTrigger),
        "back" => Some(Button::Select),
        "start" => Some(Button::Start),
        "guide" => Some(Button::Mode),
        "leftstick" => Some(Button::LeftThumb),
        "rightstick" => Some(Button::RightThumb),
        "dpup" => Some(Button::DPadUp),
        "dpdown" => Some(Button::DPadDown),
        "dpleft" => Some(Button::DPadLeft),
        "dpright" => Some(Button::DPadRight),
        _ => None,
    }
}

fn read_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;

    let mut input = String::new();
    if io::stdin().read_line(&mut input)? == 0 {
        bail!("Standard input closed");
    }

    Ok(input.trim().to_ascii_lowercase())
}

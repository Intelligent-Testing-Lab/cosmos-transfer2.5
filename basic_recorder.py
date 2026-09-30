#!/usr/bin/env python3
"""
CARLA multi-camera recorder with Traffic Manager background traffic.

Spawns:
  - One autopilot ego vehicle
  - Configurable background traffic vehicles managed by CARLA Traffic Manager
  - Configurable camera sensors attached to the ego vehicle

Records:
  - RGB / colourised depth / colourised semantic cameras -> MP4
  - Depth visualisation array -> raw_frames.npy
  - Raw semantic IDs -> raw_frames.npy

Requirements:
  pip install numpy opencv-python

Ensure CARLA's PythonAPI is importable, for example:
  export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-*.egg:$PYTHONPATH

Example:
  python record_carla_sensors.py \
      --town Town05 \
      --seconds 30 \
      --delta 0.05 \
      --traffic-vehicles 50 \
      --traffic-speed-difference 10 \
      --output-dir output/town05_run_001
"""

import argparse
import queue
import random
from pathlib import Path

import carla
import cv2
import numpy as np

# ============================================================================
# EDIT THIS LIST TO CHANGE YOUR SENSOR RIG.
#
# CARLA blueprints:
#   sensor.camera.rgb
#   sensor.camera.depth
#   sensor.camera.semantic_segmentation
#   sensor.camera.instance_segmentation
#
# Transform coordinates are relative to the ego vehicle:
#   x: forward, y: right, z: up
#
# output:
#   "mp4" -> visual stream written as MP4
#   "npy" -> numerical/label data stored in raw_frames.npy
# ============================================================================

SENSORS = [
    {
        "name": "front_rgb",
        "blueprint": "sensor.camera.rgb",
        "transform": carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        "attributes": {
            "image_size_x": "1024",
            "image_size_y": "512",
            "fov": "90",
            "sensor_tick": "0.0",
        },
        "output": "mp4",
        "converter": None,
    },
    {
        "name": "front_depth_m",
        "blueprint": "sensor.camera.depth",
        "transform": carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        "attributes": {
            "image_size_x": "1024",
            "image_size_y": "512",
            "fov": "90",
            "sensor_tick": "0.0",
        },
        "output": "npy",
        "converter": None,
    },
    {
        "name": "front_depth_log",
        "blueprint": "sensor.camera.depth",
        "transform": carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        "attributes": {
            "image_size_x": "1024",
            "image_size_y": "512",
            "fov": "90",
            "sensor_tick": "0.0",
        },
        "output": "mp4",
        "converter": carla.ColorConverter.LogarithmicDepth,
    },
    {
        "name": "front_semantic_id",
        "blueprint": "sensor.camera.semantic_segmentation",
        "transform": carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        "attributes": {
            "image_size_x": "1024",
            "image_size_y": "512",
            "fov": "90",
            "sensor_tick": "0.0",
        },
        "output": "npy",
        "converter": None,
    },
    {
        "name": "front_semantic_cityscapes",
        "blueprint": "sensor.camera.semantic_segmentation",
        "transform": carla.Transform(
            carla.Location(x=1.6, y=0.0, z=1.7),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        ),
        "attributes": {
            "image_size_x": "1024",
            "image_size_y": "512",
            "fov": "90",
            "sensor_tick": "0.0",
        },
        "output": "mp4",
        "converter": carla.ColorConverter.CityScapesPalette,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record configurable CARLA camera outputs with traffic."
    )

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8001)

    parser.add_argument("--town", default="Town03")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Fixed simulation delta in seconds; output FPS is 1 / delta.",
    )

    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--output-dir", default="output/carla_recording")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--traffic-vehicles",
        type=int,
        default=30,
        help="Number of background Traffic Manager vehicles to spawn.",
    )
    parser.add_argument(
        "--traffic-filter",
        default="vehicle.*",
        help="CARLA blueprint filter for background traffic.",
    )
    parser.add_argument(
        "--traffic-speed-difference",
        type=float,
        default=0.0,
        help=(
            "Traffic Manager global speed difference in percent. "
            "Positive values drive slower; negative values drive faster."
        ),
    )
    parser.add_argument(
        "--traffic-ignore-lights",
        type=float,
        default=0.0,
        help=("Percentage chance that background vehicles ignore traffic lights."),
    )

    return parser.parse_args()


def image_to_bgra(image: carla.Image) -> np.ndarray:
    """Copy CARLA BGRA image data into an H x W x 4 uint8 array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4)).copy()


def decode_depth_m(bgra: np.ndarray) -> np.ndarray:
    """
    Decode CARLA's 24-bit RGB depth encoding into float32 metres.

    CARLA stores normalised depth as:
      (R + 256*G + 256^2*B) / (256^3 - 1)

    The encoded depth range is 0 to 1000 metres.
    """
    b = bgra[:, :, 0].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    r = bgra[:, :, 2].astype(np.float32)

    return 1000.0 * (r + 256.0 * g + (256.0**2) * b) / ((256.0**3) - 1.0)


def make_sensor_callback(spec: dict, output_queue: queue.Queue):
    sensor_name = spec["name"]
    blueprint_name = spec["blueprint"]
    output_type = spec["output"]
    converter = spec.get("converter")

    def callback(image: carla.Image):
        if converter is not None:
            image.convert(converter)

        bgra = image_to_bgra(image)

        if output_type == "mp4":
            # Video outputs must always be 3-channel BGR.
            data = bgra[:, :, :3]

            if blueprint_name == "sensor.camera.depth" and sensor_name == "front_depth_log":
                data = 255 - data

        elif output_type == "npy":
            if blueprint_name == "sensor.camera.depth":
                data = 255 - bgra[:, :, :3]

            elif blueprint_name == "sensor.camera.semantic_segmentation":
                # Raw semantic class ID is in the red channel.
                data = bgra[:, :, 2]

            elif blueprint_name == "sensor.camera.instance_segmentation":
                semantic_id = bgra[:, :, 2].astype(np.uint32)
                instance_id = (
                    bgra[:, :, 1].astype(np.uint32)
                    + 256 * bgra[:, :, 0].astype(np.uint32)
                )
                data = np.stack((semantic_id, instance_id), axis=-1)

            else:
                data = bgra

        else:
            raise ValueError(
                f"Unsupported output type {output_type!r} "
                f"for sensor {sensor_name!r}"
            )

        output_queue.put((image.frame, sensor_name, data))

    return callback


def get_camera_resolution(spec: dict) -> tuple[int, int]:
    attributes = spec.get("attributes", {})
    width = int(attributes.get("image_size_x", 800))
    height = int(attributes.get("image_size_y", 600))
    return width, height


def spawn_ego(
    world: carla.World,
    blueprint_library,
    args,
) -> carla.Vehicle:
    """Spawn the ego vehicle at a randomly selected valid map spawn point."""
    vehicle_bp = blueprint_library.find(args.vehicle_filter)
    vehicle_bp.set_attribute("role_name", "hero")

    if vehicle_bp.has_attribute("color"):
        colours = vehicle_bp.get_attribute("color").recommended_values
        if colours:
            vehicle_bp.set_attribute("color", random.choice(colours))

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for transform in spawn_points:
        ego = world.try_spawn_actor(vehicle_bp, transform)
        if ego is not None:
            return ego

    raise RuntimeError("Unable to spawn ego vehicle at any map spawn point.")


def spawn_traffic(
    world: carla.World,
    blueprint_library,
    traffic_manager,
    args,
    ego: carla.Vehicle,
) -> list[carla.Vehicle]:
    """
    Spawn background vehicles and register them with CARLA Traffic Manager.

    Only four-wheeled blueprints are selected, avoiding bicycles and unusual
    actor types. Spawn-point selection remains seed-controlled through
    `random.seed(args.seed)` in main().
    """
    if args.traffic_vehicles <= 0:
        return []

    vehicle_blueprints = blueprint_library.filter(args.traffic_filter)
    vehicle_blueprints = [
        bp
        for bp in vehicle_blueprints
        if bp.has_attribute("number_of_wheels")
        and int(bp.get_attribute("number_of_wheels").as_int()) == 4
    ]

    if not vehicle_blueprints:
        raise RuntimeError(
            f"No four-wheeled vehicle blueprints match " f"{args.traffic_filter!r}."
        )

    ego_location = ego.get_location()
    spawn_points = world.get_map().get_spawn_points()

    # Prevent attempts immediately adjacent to the hero spawn point.
    spawn_points = [
        transform
        for transform in spawn_points
        if transform.location.distance(ego_location) > 8.0
    ]
    random.shuffle(spawn_points)

    traffic_manager.global_percentage_speed_difference(args.traffic_speed_difference)

    traffic_vehicles = []

    for transform in spawn_points:
        if len(traffic_vehicles) >= args.traffic_vehicles:
            break

        blueprint = random.choice(vehicle_blueprints)

        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")

        if blueprint.has_attribute("color"):
            colours = blueprint.get_attribute("color").recommended_values
            if colours:
                blueprint.set_attribute("color", random.choice(colours))

        if blueprint.has_attribute("driver_id"):
            drivers = blueprint.get_attribute("driver_id").recommended_values
            if drivers:
                blueprint.set_attribute(
                    "driver_id",
                    random.choice(drivers),
                )

        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            continue

        vehicle.set_autopilot(True, args.tm_port)

        traffic_manager.ignore_lights_percentage(
            vehicle,
            args.traffic_ignore_lights,
        )

        traffic_vehicles.append(vehicle)

    print(
        f"Spawned {len(traffic_vehicles)}/"
        f"{args.traffic_vehicles} background traffic vehicles."
    )

    return traffic_vehicles


def create_writer(path: Path, fps: float, width: int, height: int):
    """Create an MP4 writer using the portable mp4v codec."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {path}")

    return writer


def main():
    args = parse_args()

    if args.delta <= 0:
        raise ValueError("--delta must be greater than zero.")

    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")

    if args.traffic_vehicles < 0:
        raise ValueError("--traffic-vehicles cannot be negative.")

    if not 0.0 <= args.traffic_ignore_lights <= 100.0:
        raise ValueError("--traffic-ignore-lights must be between 0 and 100.")

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    # Starts a fresh CARLA episode in the requested map.
    world = client.load_world(args.town)
    traffic_manager = client.get_trafficmanager(args.tm_port)
    blueprint_library = world.get_blueprint_library()

    original_settings = world.get_settings()

    ego = None
    traffic_vehicles = []
    spawned_sensors = []
    writers = {}
    sensor_raw_data = {}

    sensor_by_name = {spec["name"]: spec for spec in SENSORS}

    try:
        # Fixed-step synchronous operation provides frame-aligned sensor data.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        ego = spawn_ego(world, blueprint_library, args)
        ego.set_autopilot(True, args.tm_port)

        # Keep ego at nominal Traffic Manager speed.
        traffic_manager.vehicle_percentage_speed_difference(ego, 0.0)

        traffic_vehicles = spawn_traffic(
            world=world,
            blueprint_library=blueprint_library,
            traffic_manager=traffic_manager,
            args=args,
            ego=ego,
        )

        sensor_queue = queue.Queue()

        output_fps = 1.0 / args.delta

        # Create output directories and MP4 writers.
        for spec in SENSORS:
            sensor_dir = output_dir / spec["name"]
            sensor_dir.mkdir(parents=True, exist_ok=True)

            if spec["output"] == "mp4":
                width, height = get_camera_resolution(spec)
                writers[spec["name"]] = create_writer(
                    path=sensor_dir / f"{spec['name']}.mp4",
                    fps=output_fps,
                    width=width,
                    height=height,
                )

        # Spawn and attach all configured sensors.
        for spec in SENSORS:
            blueprint = blueprint_library.find(spec["blueprint"])
            sensor_raw_data[spec["name"]] = []

            for attribute_name, attribute_value in spec.get(
                "attributes",
                {},
            ).items():
                if blueprint.has_attribute(attribute_name):
                    blueprint.set_attribute(
                        attribute_name,
                        str(attribute_value),
                    )
                else:
                    print(
                        f"Warning: {spec['blueprint']} does not expose "
                        f"attribute {attribute_name!r}; ignoring it."
                    )

            sensor = world.spawn_actor(
                blueprint,
                spec["transform"],
                attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid,
            )

            sensor.listen(make_sensor_callback(spec, sensor_queue))
            spawned_sensors.append(sensor)

        total_ticks = round(args.seconds / args.delta)
        expected_sensor_names = set(sensor_by_name)

        print(
            f"Recording {len(SENSORS)} sensors for {total_ticks} ticks "
            f"at {output_fps:.2f} FPS into:\n  {output_dir.resolve()}"
        )

        # Initialise sensors and allow Traffic Manager to begin its cycle.
        world.tick()

        for tick_index in range(total_ticks):
            world_frame = world.tick()
            received = {}

            # Sensor callback order is nondeterministic. Only retain data
            # associated with the exact world frame just advanced.
            while set(received) != expected_sensor_names:
                sensor_frame, sensor_name, data = sensor_queue.get(timeout=10.0)

                if sensor_frame == world_frame:
                    received[sensor_name] = data

            for sensor_name, data in received.items():
                spec = sensor_by_name[sensor_name]
                sensor_dir = output_dir / sensor_name

                if spec["output"] == "mp4":
                    writers[sensor_name].write(data)
                else:
                    sensor_raw_data[sensor_name].append(data)

            progress_interval = max(1, total_ticks // 10)
            if (tick_index + 1) % progress_interval == 0:
                print(f"Recorded {tick_index + 1}/{total_ticks} ticks")

    finally:
        print("Cleaning up actors and restoring world settings...")

        # Stop callback threads before persisting sensor buffers.
        for sensor in spawned_sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass

        print("Saving raw sensor data...")
        for sensor_name, raw_data in sensor_raw_data.items():
            if raw_data:
                np.save(
                    output_dir / sensor_name / "raw_frames.npy",
                    np.asarray(raw_data),
                )

        for writer in writers.values():
            writer.release()

        for sensor in spawned_sensors:
            try:
                print("Destroying Sensors")
                sensor.destroy()
            except RuntimeError:
                pass

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()

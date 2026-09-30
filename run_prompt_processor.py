import argparse
import openai
from pathlib import Path
import os

MODEL="nvidia/Cosmos-Reason1-7B"
GENERIC_PROMPT="""
Photorealistic real-world autonomous-driving camera footage with natural lighting, realistic road surfaces, detailed vehicles, buildings, vegetation, traffic infrastructure, lane markings, shadows, reflections, and physically plausible materials. High-quality forward-facing vehicle camera imagery with authentic wide-angle optics, natural exposure, subtle sensor noise, realistic motion blur, and temporally coherent driving-video appearance.
"""

SYSTEM_PROMPT="""

You are an expert at captioning various traffic scenarios for photorealistic Sim→Real style transfer.

Analyze the input video and write a comprehensive caption that will be used as the text prompt for a world foundation model. The caption must:

- Strongly emphasize photorealism and real-world appearance:
  - Begin with a clear style anchor
  - Use phrases that convey real-world materials and lighting, e.g.
  - Optionally describe camera characteristics typical of real automotive footage

While also describing all visible scene elements with physical AI expertise, including:
  - Traffic infrastructure (roads, intersections, traffic lights, lane markings)
  - Vehicles and their behaviors (including anomalies)
  - Environmental context (buildings, vegetation, urban features)
  - Current conditions (time of day, weather, visibility)
  - Lighting features (natural or artificial light, shadows, reflections)

to help maintain the semantic and temporal accuracy of the Sim to Real style transfer 

Rules:
- Describe only what is clearly visible in the video.
- Do not mention that this is part of a simulation environment or synthetic data.
- Assume any other vehicles on the road are part of normal traffic.
- If motion is important, include it briefly as visual context.
- If traffic signals are important, include it briefly as visual context.
- Do not mention that this came from a video or a simulation.
- Output only the final prompt, with no explanation or extra text.
"""

OLD_SYSTEM_PROMPT = """
You are an expert at captioning various traffic scenarions

Analyze the input video and write a comprehensive caption that decaptures all scene elements with physical AI expertise, including the following (must include): 
- Traffic infrastructure (roads, intersections, traffic lights)
- Vehicles and their behaviors (including anomalies)
- Environmental context (buildings, vegetation, urban features)
- Current conditions (time of day, weather, visibility)
- Lighting features

Rules:
- Describe only what is clearly visible in the video.
- Do not mention it is part of a simulation environment. You should aim to describe the scene in as realistic detail as possible.
- Do not invent details, identities, brands, text, or locations.
- Assume any other vehicles on the road are part of traffic.
- If something is uncertain, leave it out.
- If motion is important, include it briefly as visual context.
- Do not mention that this came from a video or a simulation.
- Output only the final prompt, with no explanation or extra text.

Example 1:
The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. The sky is clear, suggesting a sunny day, which enhances the visibility of the scene. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.

Example 2:
A scenic drive unfolds along a coastal highway. The video captures a smooth, continuous journey along a multi-lane road, with the camera positioned as if from the perspective of a vehicle traveling in the right lane. The road is bordered by a tall, green mountain on the right, which casts a shadow over part of the highway, while the left side opens up to a view of the ocean, visible in the distance beyond a row of low-lying vegetation and a sidewalk. Several vehicles, including two vehicles, travel ahead, maintaining a steady pace. The road is well-maintained, with clear white lane markings and a concrete barrier separating the lanes from the mountain covered by trees on the right. Utility poles and power lines run parallel to the road on the left, adding to the infrastructure of the scene. The camera remains static, providing a consistent view of the road and surroundings, emphasizing the serene and uninterrupted nature of the drive.

Example 3:
The video captures a nighttime urban driving scene viewed from inside a vehicle. The street is wet, reflecting the lights from the surrounding buildings and street lamps, creating a moody and atmospheric setting. The camera, positioned inside the vehicle, provides a first-person perspective as the car moves forward. The road is lined with tall, multi-story buildings, their windows illuminated, suggesting it's either late evening or early morning. As the vehicle progresses, the camera passes by parked cars on both sides of the street, including a white SUV and a black Jeep. The street is relatively quiet, with minimal pedestrian activity, and the occasional passing car can be seen in the distance. The traffic lights ahead are green, allowing the vehicle to continue moving smoothly. The wet pavement reflects the lights, adding a shimmering effect to the scene. The environment is characterized by a mix of residential and commercial buildings, with some shops visible along the sidewalk. The overall ambiance is calm and serene, with the soft glow of the streetlights and building lights providing a sense of safety and order. The camera remains steady throughout, capturing the continuous forward motion of the vehicle as it navigates through this urban landscape.
"""

def check_file(path: str): 
    path_obj = Path(path)
    return path_obj.exists()

def get_path(path: str):
    return Path(path)

def generate_prompt(client, input_video) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (SYSTEM_PROMPT)
            }, 
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a single grounded diffusion prompt for sim2real transfer for this video."
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"file://{input_video}"
                        }
                    }
                ]
            }
        ],
        max_tokens=5200,
        temperature=0.0,
    )
    return resp.choices[0].message.content

def main():
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )
    
    output_dir = get_path(os.getenv("OUTPUT_DIR", ""))
    output_dir.mkdir(parents=True, exist_ok=True)

    video_input = get_path(os.getenv("RGB_INPUT", ""))
    try:
        diffusion_prompt = generate_prompt(client, video_input)
    except Exception as e:
        print("Generation failed: using generic prompt")
        diffusion_prompt = GENERIC_PROMPT

    with open(output_dir / "prompt.txt", "w") as f:
        f.write(diffusion_prompt)

if __name__ == '__main__':
    main()

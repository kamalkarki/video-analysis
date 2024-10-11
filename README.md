# Project Video Analysis 

This project demonstrates an end-to-end pipeline for analyzing videos specially survillance videos using vision and text based Llms.
The pipeline involves frame extraction (based on the length of the video), vision language models (LLMs) for generating text descriptions, and natural language processing tasks like text summarization and entity extraction.

## Inspiration
We had a incident at one of our relative home, it was a home survillance camera and we spent quite some time to find a red car which passed from the front gate. So I thought of building a system to anayze videos and extract information using GEN AI tools.

## Project Overview

The goal of this project is to automate the process of analyzing surveillance footage by breaking down a video into individual frames, describing the content of each frame, and summarizing the overall activity with relevant entity extraction for specific objects or events. The major steps involved are:

1. **Frame Extraction**: 
   - Extract a frame from the video for every second of footage using openCV
   
2. **Frame Description Generation**:
   - For each extracted frame, a text description is generated using a pre-trained vision-language model from Hugging Face 
   ex - qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8 and microsoft/Phi-3.5-vision-instruct
   
3. **Text Summarization**:
   - Summarize the generated text descriptions to provide an overview of the events in the video.
   ex - Qwen2.5-3B-Instruct and Llama 3.2 Instruct 3B
4. **Entity Extraction**:
   - Extract key entities from the summarized text, such as vehicle people, vehicles[type, color].

## Workflow

1. **Video Input**:
   - A surveillance video is loaded into the system. The video is processed and divided into frames, one frame per second ( this can be configure based on the length of the video)
   
2. **Frame Extraction**:
   - Frames are extracted at 1-second intervals using `OpenCV` or other video processing libraries.

3. **Description Generation**:
   - Each extracted frame is passed through a vision-language model to generate a detailed text description of the content in that frame.

4. **Summarization**:
   - After descriptions are generated for each frame, the descriptions are aggregated and summarized using a pre-trained text summarization model to give an overall description of the scene.

5. **Entity Extraction**:
   - The summarized text is further processed using an entity extraction model to identify key objects, people, or events, such as traffic lights, vehicles, pedestrians, and potential incidents (e.g., collisions).

## Prerequisites

- Python 3.x
- Hugging Face Transformers
- OpenCV (for video and frame processing)
- PyTorch or TensorFlow (for model inference)
- Please refer requirements file for further details


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kamal/video-analysis-transformer.git
   cd video-analysis-transformer
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Models are downloaded and kept in the models directory in the code base

## Usage
To run this project, you need to run two python scripts.

1. app.py
2. gradio_app.py
3. once you run the gradio app please select the video from the "analysis\data\input\video" folder and click on submit.
4. once the video is processed you will see the summary and entities in the UI.


Given a surveillance video of an intersection, the pipeline will:

1. Extract frames at each second.
2. Generate text descriptions, such as:
   
   - "A blue car is approaching the intersection."
   - "A pedestrian is crossing the road."
   - "A red car runs through a red light."

3. Summarize the descriptions to something like:
   
   - "The video shows a busy intersection with multiple vehicles and pedestrians. A red car runs a red light and causes a collision."

4. Extract entities such as:
   
   - **Vehicle**: Red car, blue car
   - **Event**: Collision
   - **Traffic Light**: Red light

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/LICENSE.txt)

This project may download and install additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 

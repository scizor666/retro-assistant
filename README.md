# Retro Assistant App

Retro Assistant is an Android application that provides a chat interface for interacting with on-device GGUF models using the Arm AI Chat inference engine.

## Model Requirements

**IMPORTANT**: To use this application, you must provide a GGUF model file. The app does not come with a pre-loaded model. You can select a model file from your device's storage in the **Settings** screen.

### Recommended Model

For the best experience, we recommend using our fine-tuned model, which has been optimized for this application. You can download it from the following link:

[Download Fine-tuned Retro GGUF Model](https://drive.google.com/file/d/1guZTLwyqjkNMVrjYcoiq6j0-c4XF0t0Q/view?usp=drive_link)

This model was trained and exported using the tools and scripts available in our companion repository: [retro-model-tuning](https://github.com/scizor666/retro-model-tuning).

## Getting Started

1.  Build and install the application on your Android device.
2.  Download the recommended GGUF model file (or provide your own).
3.  Open the app and navigate to **Settings** (gear icon in the top right).
4.  Tapped **Change GGUF Model** and select the downloaded `.gguf` file.
5.  Wait for the model to be processed and loaded (status indicators will guide you).
6.  Start chatting!

## Features

- **1000 Conversation History**: Persistent storage for up to 1000 chat sessions.
- **Deep Model Integration**: Direct interaction with on-device GGUF models.
- **Data Management**: Easily clear history or reset all app data from Settings.
- **Modern UI**: Clean, responsive layout with dark mode support and clear status feedback.

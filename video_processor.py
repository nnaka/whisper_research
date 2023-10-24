from pytube import YouTube
from moviepy.editor import VideoFileClip, AudioFileClip
from typing import Optional


def download_youtube_video(
    video_url: str, save_path: str, save_name: Optional[str] = None
) -> str:
    """
    Download a YouTube video to the specified path.

    Args:
        video_url (str): The URL of the YouTube video to download.
        save_path (str): The path where the video will be saved.
    """
    yt: YouTube = YouTube(video_url)
    video_stream = yt.streams.get_highest_resolution()
    filename: str = save_name if save_name is not None else yt.title
    print(f"Downloading video: {yt.title} to {save_path}/{filename}")
    video_stream.download(output_path=save_path, filename=filename)
    return f"{save_path}/{filename}"


def separate_audio(video_path: str, audio_output_path: str) -> None:
    """
    Separate the audio from a video and save it as a separate audio file.

    Args:
        video_path (str): The path to the video file.
        audio_output_path (str): The path where the audio will be saved.
    """
    video_clip: VideoFileClip = VideoFileClip(video_path)
    audio_clip: AudioFileClip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)
    video_clip.close()
    audio_clip.close()
    print(f"Audio extracted and saved as {audio_output_path}")


if __name__ == "__main__":
    # Replace with the YouTube video URL you want to download
    # video_url: str = "https://www.youtube.com/watch?v=your_video_id_here"
    video_url: str = "https://youtu.be/aPBZx1Kpypc"

    # Define the paths for saving the video and audio
    video_save_path: str = "/scratch/nn1331/whisper/video"
    audio_output_path: str = "/scratch/nn1331/whisper/audio/audio.mp3"

    # Download the YouTube video
    video_file_path: str = download_youtube_video(video_url, video_save_path, "test.mp4")

    # Separate the audio from the video
    separate_audio(video_file_path, audio_output_path)

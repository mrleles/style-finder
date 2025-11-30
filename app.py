"""
Main application file for the Style Finder Gradio interface.
"""

import gradio as gr
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import local modules
from models.image_processor import ImageProcessor
from models.llm_service import LlamaVisionService
from utils.helpers import get_all_items_for_image, format_alternatives_response, process_response
import config

class StyleFinderApp:
    """
    Main application class that orchestrates the Style Finder workflow.
    """
    
    def __init__(self, dataset_path, serp_api_key=None):
        """
        Initialize the Style Finder application.
        
        Args:
            dataset_path (str): Path to the dataset file
            serp_api_key (str, optional): SerpAPI key for product searches
            
        Raises:
            FileNotFoundError: If the dataset file is not found
            ValueError: If the dataset is empty or invalid
        """
        # Check if dataset file exists and raise FileNotFoundError if not
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        # Load the dataset
        self.data = pd.read_pickle(dataset_path)
        
        # Check if dataset is empty and raise ValueError if it is
        if self.data.empty:
            raise ValueError("The loaded dataset is empty")
        
        # Initialize image processor component
        self.image_processor = ImageProcessor(
            image_size=config.IMAGE_SIZE,
            norm_mean=config.NORMALIZATION_MEAN,
            norm_std=config.NORMALIZATION_STD
        )
        
        # Initialize LLM service component
        self.llm_service = LlamaVisionService(
            model_id=config.LLAMA_MODEL_ID,
            project_id=config.PROJECT_ID,
            region=config.REGION
        )
        
        # TODO: Initialize search service component if API key is provided

    def process_image(self, image):
        """
        Process a user-uploaded image and generate a fashion response.
        
        Args:
            image: PIL image uploaded through Gradio
                
        Returns:
            str: Formatted response with fashion analysis
        """
        # Save the image temporarily if it's not already a file path
        if not isinstance(image, str):
            temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
            image_path = temp_file.name
            image.save(image_path)
        else:
            image_path = image
        
        # Encode the image using the image processor
        user_encoding = self.image_processor.encode_image(image_path, is_url=False)
        
        # Check if encoding was successful
        if user_encoding['vector'] is None:
            return "Error: Unable to process the image. Please try another image."
        
        # Find the closest match in the dataset
        closest_row, similarity_score = self.image_processor.find_closest_match(user_encoding['vector'], self.data)
        
        # Check if a match was found and Log match details
        if closest_row is None:
            return "Error: Unable to find a match. Please try another image."
        
        # Get all related items for the matched image
        all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
        
        # Check if items were found
        if all_items.empty:
            return "Error: No items found for the matched image."
        
        # Generate fashion response using the LLM service
        bot_response = self.llm_service.generate_fashion_response(
            user_image_base64=user_encoding['base64'],
            matched_row=closest_row,
            all_items=all_items,
            similarity_score=similarity_score,
            threshold=config.SIMILARITY_THRESHOLD
        )
        
        # Clean up temporary files
        if not isinstance(image, str):
            try:
                os.unlink(image_path)
            except:
                pass
        
        # Process and return the response
        return process_response(bot_response)


def create_gradio_interface(app):
    """
    Create and configure the Gradio interface.
    
    Args:
        app (StyleFinderApp): Instance of the StyleFinderApp
        
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Create Gradio Blocks interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Analyzer") as demo:
    
        # Add introduction section
        gr.Markdown(
            """
                # Fashion Style Analyzer
                
                Upload an image to analyze fashion elements and get detailed information about the items.
                This application combines computer vision, vector similarity, and large language models 
                to provide detailed fashion analysis.
                """
        )
        
        # Add example images section
        gr.Markdown('### Example Images')
        with gr.Row():
            gr.Image(value="examples/test-1.png", label="Example 1", show_label=True, scale=1)
            gr.Image(value="examples/test-2.png", label="Example 2", show_label=True, scale=1)
            gr.Image(value="examples/test-3.png", label="Example 3", show_label=True, scale=1)
        
        # Add example image buttons
        with gr.Row():
            example1_btn = gr.Button("Use Example 1")
            example2_btn = gr.Button("Use Example 2")
            example3_btn = gr.Button("Use Example 3")
        
        # Add image input, submit button, and status components
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Fashion Image"
                )
    
                submit_btn = gr.Button("Analyze Style", variant="primary")
    
                status = gr.Markdown("Ready to analyze.")
        
        # Add output display component
        with gr.Column(scale=2):
            output = gr.Markdown(
                label="Style Analysis Results",
                height=700
            )
        
        # Configure submit button click event handlers
        submit_btn.click(
            fn=lambda: "Analyzing image... This may take a few moments.",
            inputs=None,
            outputs=status
        ).then(
            fn=app.process_image,
            inputs=[image_input],
            outputs=output
        ).then(
            fn=lambda: "Analysis complete!",
            inputs=None,
            outputs=status
        )
        
        # Configure example image button event handlers
        example1_btn.click(
            fn=lambda: "examples/test-1.png",
            inputs=None,
            outputs=image_input
        ).then(
            fn=lambda: "Example 1 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
    
        example2_btn.click(
            fn=lambda: "examples/test-2.png",
            inputs=None,
            outputs=image_input
        ).then(
            fn=lambda: "Example 2 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
    
        example3_btn.click(
            fn=lambda: "examples/test-3.png",
            inputs=None,
            outputs=image_input
        ).then(
            fn=lambda: "Example 3 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status
        )
        
        # Add information about the application
        gr.Markdown(
            """
            ### about This Application
    
            This system analyzes fashion images using:
    
            - **Image Encoding**: Converting fashion images into numerical vectors
            - **Similarity Matching**: Finding visually similar items in a database
            - **Advanced AI**: Generating detailed descriptions of fashion elements
    
            The analyzer identifies garments, fabrics, colors, and styling details from images.
            The database includes information on outfits with brand and pricing details
            """
        )
        
        # Return the configured interface
        return demo

if __name__ == "__main__":
    try:
        # Initialize the app with the dataset
        app = StyleFinderApp("swift-style-embeddings.pkl")
        
        # Create the Gradio interface
        demo = create_gradio_interface(app)
        
        # Launch the Gradio interface
        demo.launch(
            server_name="127.0.0.1",  
            server_port=5000,
            share=True  # Set to False if you don't want to create a public link
        )
    except Exception as e:
        print(f"Error starting the application: {str(e)}") 

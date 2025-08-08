import os
from text_summarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self, example_batch):
        """
        Convert text examples to tokenized features.
        This method tokenizes the article and highlights, then returns only the tokenized fields.
        """
        try:
            # Tokenize the input articles
            input_encodings = self.tokenizer(
                example_batch['article'], 
                max_length=1024, 
                truncation=True,
                padding=True,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Tokenize the target highlights
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch['highlights'], 
                    max_length=128, 
                    truncation=True,
                    padding=True,
                    return_tensors=None  # Return lists, not tensors
                )
            
            # Return only the tokenized fields, removing the original text fields
            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except Exception as e:
            logger.error(f"Error in convert_examples_to_features: {e}")
            raise e
    

    def convert(self):
        """
        Convert the dataset by tokenizing articles and highlights.
        """
        try:
            # Load the dataset from CSV files
            data_path = os.path.join(self.config.data_path, "cnn_dailymail")
            logger.info(f"Loading dataset from {data_path}")
            
            newspaper_text_summarizer = load_dataset('csv', data_files={
                'train': os.path.join(data_path, 'train.csv'),
                'test': os.path.join(data_path, 'test.csv'),
                'validation': os.path.join(data_path, 'validation.csv')
            })
            
            logger.info("Dataset loaded successfully. Starting tokenization...")
            
            # Apply tokenization and remove original text fields
            newspaper_text_summarizer_pt = newspaper_text_summarizer.map(
                self.convert_examples_to_features, 
                batched=True,
                remove_columns=['article', 'highlights']  # Remove original text fields
            )
            
            # Save the processed dataset
            save_path = os.path.join(self.config.root_dir, "newspaper_dataset")
            newspaper_text_summarizer_pt.save_to_disk(save_path)
            logger.info(f"Processed dataset saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error in convert method: {e}")
            raise e


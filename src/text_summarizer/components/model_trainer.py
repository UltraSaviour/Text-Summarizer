from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import ModelTrainerConfig
import torch
import os
from text_summarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer from {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        
        logger.info(f"Loading model from {self.config.model_ckpt}")
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        
        # Move model to device
        model_pegasus = model_pegasus.to(device)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model_pegasus.config.pad_token_id = tokenizer.pad_token_id
        
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Loading data 
        logger.info(f"Loading dataset from {self.config.data_path}")
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Dataset path {self.config.data_path} does not exist")
            
        newspaper_text_summarizer_pt = load_from_disk(self.config.data_path)
        logger.info(f"Dataset loaded successfully. Available splits: {list(newspaper_text_summarizer_pt.keys())}")
        
        # Remove the 'id' field from all splits to avoid tensor creation issues
        for split in newspaper_text_summarizer_pt.keys():
            if 'id' in newspaper_text_summarizer_pt[split].column_names:
                newspaper_text_summarizer_pt[split] = newspaper_text_summarizer_pt[split].remove_columns(['id'])
                logger.info(f"Removed 'id' field from {split} split")
        
        # Check dataset structure and ensure only tokenized fields are present
        train_sample = newspaper_text_summarizer_pt["validation"][0]  # Using validation as training set
        logger.info(f"Train sample keys (using validation set): {list(train_sample.keys())}")
        
        # Verify that we have the expected tokenized fields
        expected_fields = ['input_ids', 'attention_mask', 'labels']
        for field in expected_fields:
            if field not in train_sample:
                raise ValueError(f"Expected field '{field}' not found in dataset. Available fields: {list(train_sample.keys())}")
        
        logger.info(f"Input IDs length: {len(train_sample['input_ids'])}")
        logger.info(f"Labels length: {len(train_sample['labels'])}")
        logger.info(f"Training on validation set ({len(newspaper_text_summarizer_pt['validation'])} samples)")
        logger.info(f"Evaluating on test set ({len(newspaper_text_summarizer_pt['test'])} samples)")

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy, 
            eval_steps=self.config.eval_steps, 
            save_steps=1e6,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Add this to avoid potential memory issues
            fp16=True,  # Enable mixed precision training to reduce memory usage
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            optim="adamw_torch"  # Use PyTorch's AdamW optimizer
        ) 

        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=newspaper_text_summarizer_pt["validation"], 
            eval_dataset=newspaper_text_summarizer_pt["test"]
        )
        
        logger.info("Starting training...")
        trainer.train()

        ## Save model
        model_save_path = os.path.join(self.config.root_dir,"pegasus-newspaper-model")
        logger.info(f"Saving model to {model_save_path}")
        model_pegasus.save_pretrained(model_save_path)
        
        ## Save tokenizer
        tokenizer_save_path = os.path.join(self.config.root_dir,"tokenizer")
        logger.info(f"Saving tokenizer to {tokenizer_save_path}")
        tokenizer.save_pretrained(tokenizer_save_path)
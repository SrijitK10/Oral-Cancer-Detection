"""
Configuration utilities for the Oral Cancer Classification Pipeline.
"""

import os
import yaml
import argparse
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Oral Cancer Classification Pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yml", 
        help="Path to the configuration YAML file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "predict"], 
        default="train", 
        help="Pipeline mode: train, evaluate, or predict"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model name to use for evaluation or prediction"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Override dataset directory from config"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--no_augment", 
        action="store_true", 
        help="Disable data augmentation"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override number of training epochs from config"
    )
    
    return parser.parse_args()


def merge_configs(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Merge configuration from YAML and command-line arguments.
    Command-line arguments take precedence over YAML configuration.
    
    Args:
        config: Configuration dictionary from YAML
        args: Command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Override dataset directory
    if args.data_dir:
        config["dataset"]["base_dir"] = args.data_dir
    
    # Override batch size
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    
    # Override epochs
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    # Override augmentation
    if args.no_augment:
        config["augmentation"]["enabled"] = False
    else:
        # Ensure enabled key exists
        config["augmentation"]["enabled"] = config["augmentation"].get("enabled", True)
    
    return config
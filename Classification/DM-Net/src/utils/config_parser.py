import os
import yaml
import argparse
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to override base_config
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    def _merge_dicts(base, override):
        for key, value in override.items():
            if key in base and isinstance(value, dict) and isinstance(base[key], dict):
                _merge_dicts(base[key], value)
            else:
                base[key] = value
    
    _merge_dicts(merged_config, override_config)
    return merged_config


def parse_cli_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="DM-Net Oral Cancer Classification")
    
    # Config file arguments
    parser.add_argument('--config', type=str, default='config/default.yml',
                        help='Path to the base configuration file')
    parser.add_argument('--train-config', type=str, default='config/train_config.yml',
                        help='Path to the training configuration file')
    parser.add_argument('--test-config', type=str, default='config/test_config.yml',
                        help='Path to the testing configuration file')
    
    # Mode argument
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], 
                        default='train', help='Mode to run the model in')
    
    # Common override arguments
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--checkpoint-path', type=str, help='Override checkpoint path')
    
    # Training override arguments
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--optimizer', type=str, help='Override optimizer')
    
    # Model override arguments
    parser.add_argument('--model', type=str, help='Override model architecture')
    parser.add_argument('--num-classes', type=int, help='Override number of classes')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='Device to use for training/testing')
    
    return parser.parse_args()


def get_config_from_args(args):
    """
    Build configuration from command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Final configuration dictionary
    """
    # Load base configuration
    config = load_yaml_config(args.config)
    
    # Load mode-specific configuration
    if args.mode == 'train' and os.path.exists(args.train_config):
        train_config = load_yaml_config(args.train_config)
        config = merge_configs(config, train_config)
    elif args.mode == 'test' and os.path.exists(args.test_config):
        test_config = load_yaml_config(args.test_config)
        config = merge_configs(config, test_config)
    
    # Override with CLI arguments
    cli_overrides = {}
    
    # Data overrides
    if args.data_dir:
        cli_overrides.setdefault('data', {})['data_dir'] = args.data_dir
    if args.batch_size:
        cli_overrides.setdefault('data', {})['batch_size'] = args.batch_size
        
    # Model overrides
    if args.model:
        cli_overrides.setdefault('model', {})['architecture'] = args.model
    if args.num_classes:
        cli_overrides.setdefault('model', {})['num_classes'] = args.num_classes
    if args.checkpoint_path:
        cli_overrides.setdefault('model', {})['checkpoint_path'] = args.checkpoint_path
        
    # Training overrides
    if args.epochs:
        cli_overrides.setdefault('training', {})['num_epochs'] = args.epochs
    if args.lr:
        cli_overrides.setdefault('training', {})['learning_rate'] = args.lr
    if args.optimizer:
        cli_overrides.setdefault('training', {})['optimizer'] = args.optimizer
        
    # Merge CLI overrides with config
    if cli_overrides:
        config = merge_configs(config, cli_overrides)
    
    # Add runtime settings
    config['runtime'] = {
        'mode': args.mode,
        'seed': args.seed,
        'device': args.device
    }
    
    return config
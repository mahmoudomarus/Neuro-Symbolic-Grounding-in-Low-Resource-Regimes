"""
Tests for HuggingFace integration and automatic upload.

Tests cover:
- Model card generation
- Upload functionality (with mocking)
- HuggingFaceCallback
- Environment variable token handling
"""

import pytest
import torch
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelCard:
    """Test model card generation."""
    
    def test_create_model_card_content(self):
        """Test model card content generation."""
        from scripts.train_multimodal import create_model_card_content
        
        metrics = {
            'val_loss': 0.1234,
            'epochs': 50,
        }
        
        content = create_model_card_content(metrics)
        
        # Check required sections
        assert "NSCA Cross-Modal World Model" in content
        assert "license: mit" in content
        assert "0.1234" in content
        assert "50" in content
        assert "Hierarchical Fusion" in content
    
    def test_create_model_card_file(self):
        """Test model card file creation."""
        from scripts.train_multimodal import create_model_card
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = {'val_loss': 0.5, 'epochs': 10}
            
            card_path = create_model_card(tmpdir, metrics)
            
            assert Path(card_path).exists()
            
            with open(card_path, 'r') as f:
                content = f.read()
            
            assert "NSCA" in content


class TestUploadFunction:
    """Test upload functionality."""
    
    @patch('scripts.train_multimodal.HfApi')
    @patch('scripts.train_multimodal.login')
    def test_upload_with_token(self, mock_login, mock_api_class):
        """Test upload with provided token."""
        from scripts.train_multimodal import upload_to_huggingface
        
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'test': 'data'}, f.name)
            model_path = f.name
        
        try:
            result = upload_to_huggingface(
                model_path=model_path,
                repo_id="test/repo",
                hf_token="test_token_123",
                commit_message="Test upload",
            )
            
            # Should call login and upload
            mock_login.assert_called_once_with(token="test_token_123")
            mock_api.upload_file.assert_called_once()
            assert result == True
        finally:
            os.unlink(model_path)
    
    @patch.dict(os.environ, {'HF_TOKEN': 'env_token_456'})
    @patch('scripts.train_multimodal.HfApi')
    @patch('scripts.train_multimodal.login')
    def test_upload_with_env_token(self, mock_login, mock_api_class):
        """Test upload using environment variable token."""
        from scripts.train_multimodal import upload_to_huggingface
        
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'test': 'data'}, f.name)
            model_path = f.name
        
        try:
            result = upload_to_huggingface(
                model_path=model_path,
                repo_id="test/repo",
                hf_token=None,  # Should use env var
            )
            
            mock_login.assert_called_once_with(token="env_token_456")
            assert result == True
        finally:
            os.unlink(model_path)
    
    def test_upload_without_token(self):
        """Test upload fails gracefully without token."""
        from scripts.train_multimodal import upload_to_huggingface
        
        # Clear any env token
        with patch.dict(os.environ, {}, clear=True):
            result = upload_to_huggingface(
                model_path="/fake/path.pth",
                repo_id="test/repo",
                hf_token=None,
            )
            
            assert result == False
    
    @patch('scripts.train_multimodal.HfApi')
    @patch('scripts.train_multimodal.login')
    def test_upload_with_version(self, mock_login, mock_api_class):
        """Test upload with version string."""
        from scripts.train_multimodal import upload_to_huggingface
        
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'test': 'data'}, f.name)
            model_path = f.name
        
        try:
            upload_to_huggingface(
                model_path=model_path,
                repo_id="test/repo",
                hf_token="token",
                version="1.0",
            )
            
            # Check that version is in the path
            call_args = mock_api.upload_file.call_args
            assert "v1.0" in call_args.kwargs['path_in_repo']
        finally:
            os.unlink(model_path)


class TestHuggingFaceCallback:
    """Test the HuggingFaceCallback class."""
    
    def test_callback_initialization(self):
        """Test callback initializes correctly."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
            callback = HuggingFaceCallback(
                repo_id="test/repo",
                upload_best_only=True,
            )
            
            assert callback.enabled == True
            assert callback.repo_id == "test/repo"
            assert callback.upload_best_only == True
    
    def test_callback_disabled_without_token(self):
        """Test callback disables without token."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with patch.dict(os.environ, {}, clear=True):
            callback = HuggingFaceCallback(
                repo_id="test/repo",
                token=None,
            )
            
            assert callback.enabled == False
    
    @patch('scripts.train_multimodal.upload_to_huggingface')
    def test_callback_on_epoch_end_improvement(self, mock_upload):
        """Test callback uploads on improvement."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        mock_upload.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model file
            model_path = Path(tmpdir) / "model.pth"
            torch.save({'test': 'data'}, model_path)
            
            with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
                callback = HuggingFaceCallback(
                    repo_id="test/repo",
                    upload_best_only=True,
                    save_dir=tmpdir,
                )
                
                train_metrics = {'total_loss': 0.5}
                val_metrics = {'total_loss': 0.4}
                
                result = callback.on_epoch_end(
                    epoch=0,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    model_path=str(model_path),
                )
                
                assert result == True
                mock_upload.assert_called()
    
    @patch('scripts.train_multimodal.upload_to_huggingface')
    def test_callback_no_upload_without_improvement(self, mock_upload):
        """Test callback doesn't upload without improvement."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            torch.save({'test': 'data'}, model_path)
            
            with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
                callback = HuggingFaceCallback(
                    repo_id="test/repo",
                    upload_best_only=True,
                    save_dir=tmpdir,
                )
                
                # First epoch - should upload
                callback.best_val_loss = 0.3
                
                val_metrics = {'total_loss': 0.5}  # Worse than best
                
                result = callback.on_epoch_end(
                    epoch=1,
                    train_metrics={'total_loss': 0.5},
                    val_metrics=val_metrics,
                    model_path=str(model_path),
                )
                
                assert result == False
                mock_upload.assert_not_called()


class TestTokenHandling:
    """Test secure token handling."""
    
    def test_env_token_priority(self):
        """Test that env token is used when no explicit token provided."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with patch.dict(os.environ, {'HF_TOKEN': 'env_token'}):
            callback = HuggingFaceCallback(
                repo_id="test/repo",
                token=None,  # Should use env
            )
            
            assert callback.token == 'env_token'
    
    def test_explicit_token_priority(self):
        """Test that explicit token overrides env token."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with patch.dict(os.environ, {'HF_TOKEN': 'env_token'}):
            callback = HuggingFaceCallback(
                repo_id="test/repo",
                token='explicit_token',
            )
            
            assert callback.token == 'explicit_token'
    
    def test_token_not_logged(self):
        """Test that token is not logged or printed."""
        from scripts.train_multimodal import HuggingFaceCallback
        import io
        from contextlib import redirect_stdout
        
        with patch.dict(os.environ, {'HF_TOKEN': 'secret_token_123'}):
            f = io.StringIO()
            with redirect_stdout(f):
                callback = HuggingFaceCallback(
                    repo_id="test/repo",
                )
            
            output = f.getvalue()
            assert 'secret_token_123' not in output


class TestIntegration:
    """Integration tests for the full upload pipeline."""
    
    def test_training_with_callback_mock(self):
        """Test training integration with mocked upload."""
        from scripts.train_multimodal import HuggingFaceCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
                callback = HuggingFaceCallback(
                    repo_id="test/repo",
                    upload_best_only=True,
                    save_dir=tmpdir,
                )
                
                # Simulate training loop
                model_path = Path(tmpdir) / "model.pth"
                torch.save({}, model_path)
                
                with patch('scripts.train_multimodal.upload_to_huggingface') as mock_upload:
                    mock_upload.return_value = True
                    
                    # Epoch 1 - improvement
                    callback.on_epoch_end(
                        epoch=0,
                        train_metrics={'total_loss': 0.5},
                        val_metrics={'total_loss': 0.4},
                        model_path=str(model_path),
                    )
                    
                    # Epoch 2 - improvement
                    callback.on_epoch_end(
                        epoch=1,
                        train_metrics={'total_loss': 0.4},
                        val_metrics={'total_loss': 0.3},
                        model_path=str(model_path),
                    )
                    
                    # Epoch 3 - no improvement
                    callback.on_epoch_end(
                        epoch=2,
                        train_metrics={'total_loss': 0.35},
                        val_metrics={'total_loss': 0.35},
                        model_path=str(model_path),
                    )
                    
                    # Should have uploaded twice (epochs 0 and 1)
                    assert mock_upload.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

from diabetes_prediction.dataset import get_dataloader

def test_get_dataloader():
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0
    assert len(test_dataloader) > 0
try:
    import torchdata
    print(f"torchdata version: {torchdata.__version__}")
    from torchdata.datapipes.iter import IterDataPipe
    print("Successfully imported IterDataPipe")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

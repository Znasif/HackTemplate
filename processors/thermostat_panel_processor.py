from .region_processor import RegionIOProcessor

processor = RegionIOProcessor(folder_path = "./models/cars/panel2_output/")
app = processor.app
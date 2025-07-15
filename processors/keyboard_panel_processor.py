from .region_processor import RegionIOProcessor

processor = RegionIOProcessor(folder_path = "./models/cars/panel4_output/")
app = processor.app
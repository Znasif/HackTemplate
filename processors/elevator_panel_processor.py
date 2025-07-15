from .region_processor import RegionIOProcessor

processor = RegionIOProcessor(folder_path = "./models/cars/panel3_output/")
app = processor.app
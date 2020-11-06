from processors.mix_retina_anchor_free import DDPApexProcessor
# from processors.retina_anchor_free import DDPApexProcessor

# python -m torch.distributed.launch --nproc_per_node=4 main.py

if __name__ == '__main__':
    processor = DDPApexProcessor(cfg_path="config/retina_free_anchor.yaml")
    processor.run()

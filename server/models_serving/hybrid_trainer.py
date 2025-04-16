# server/hybrid_trainer.py
class HybridTrainer:
    def __init__(self):
        self.static_model = load_pretrained()
        self.dynamic_model = TemporalAttention()
        self.optimizer = HybridOptimizer()
        
    def update_parameters(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 32)
        self.lr = kwargs.get('lr', 1e-4)
        
    def hybrid_loss(self, static_out, dynamic_out, targets):
        static_loss = F.binary_cross_entropy(static_out, targets)
        dynamic_loss = F.mse_loss(dynamic_out, targets)
        return 0.7*static_loss + 0.3*dynamic_loss
    
    def run_epoch(self):
        for batch in self.dataloader:
            static_pred = self.static_model(batch)
            dynamic_pred = self.dynamic_model(batch)
            
            loss = self.hybrid_loss(
                static_pred, 
                dynamic_pred, 
                batch['labels']
            )
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

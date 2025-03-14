import os
import torch
import src.models as models
import torch.nn as nn
import random
from tqdm.auto import tqdm
from src.saver import Saver
from src.utils import SupConLoss, MultiPosConLoss
from sklearn import metrics
from src.utils import EEGToMelSpectrogram, FocalLoss,compute_mi_cfm,compute_pcc_cfm,compute_cross_entropy,compute_plv_cfm,compute_te_cfm,fuse_cfms # augment_features
from src.augmentations import gaussian_noise, ft_surrogate
import torch.nn.functional as F

class Trainer:

    def __init__(self, args):
        # Store args
        self.args = args
        # Create criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # self.criterion = FocalLoss()
        # self.contrastive_criterion = SupConLoss()
        # Create saver
        if not args.inference:
            self.saver = Saver(args.logdir, args.tag)
        
    def windowing(self, batch):
        "Use for validation and test"
        data = batch['eeg']
        
        # divide in chunks according to the model (crop size 1000)
        n_chunks = data.shape[2] // self.args.crop_size
        chunks = torch.split(data, self.args.crop_size, dim=2)
        include_last = (data.shape[2] % self.args.crop_size) == 0
        if include_last:
            chunks = torch.cat(chunks, dim=0)
        else:
            chunks = torch.cat(chunks[:-1], dim=0)
        bs = chunks.shape[0]
        assert bs == n_chunks, f"Batch size {bs} different from number of chunks {n_chunks}"
        
        batch['eeg'] = chunks
        batch['label'] = batch['label'].repeat(n_chunks)

        cfms = []
        for chunk in chunks:
            
            cfm_1 = compute_pcc_cfm(chunk)  # Convert EEG to mel-spectrogram
            cfm_2 = compute_mi_cfm(chunk)
            cfm = fuse_cfms(cfm_1,cfm_2)
            cfms.append(cfm)
        
        # Concatenate all mel-spectrograms and add them to the batch
        batch['cfm'] = torch.stack(torch.tensor(cfms,dtype=torch.float32))
        
        return batch
        
    def train(self, loaders):

        # Compute splits names
        splits = list(loaders.keys())

        # Setup model
        module = getattr(models, self.args.model)
        net = getattr(module, "Model")(vars(self.args))
        
        # Check resume
        if self.args.resume is not None:
            net.load_state_dict(Saver.load_state_dict(self.args.resume))

        # Move to device
        net.to(self.args.device)

        # Optimizer params
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
        
        # Create optimizer
        optim_class = getattr(torch.optim, self.args.optimizer)
        optim = optim_class(params=[param for param in net.parameters() if param.requires_grad], **optim_params)

        # Configure scheduler
        if self.args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optim, 
                mode = 'min', 
                patience=self.args.patience, 
                factor=self.args.reduce_lr_factor
            )
        else:
            scheduler = None

        # Initialize the final result metrics
        result_metrics = { split: {} for split in splits }

        # Train metrics
        lowest_train_loss = float('inf')
        
        # Validation metrics
        max_val_accuracy = -1
        max_val_accuracy_balanced = -1
        
        # Watch model if enabled
        if self.args.watch_model == True:
            self.saver.watch_model(net)

        # Process each epoch
        try:
            
            for epoch in range(self.args.epochs):
                
                # Process each split
                for split in splits:
            
                    # Epoch metrics
                    epoch_metrics = {}
                    epoch_labels = []
                    epoch_outputs = []

                    # Set network mode
                    if split == 'train':
                        net.train()
                        torch.set_grad_enabled(True)
                    elif epoch >= self.args.eval_after:
                        net.eval()
                        torch.set_grad_enabled(False)
                    else:
                        break
                    
                    # Process each batch
                    for batch in tqdm(loaders[split]):
                        
                        # Use windowing for validation but not for cfm
                        # if split != 'train':
                        #     batch = self.windowing(batch)
                            
                        # Get inputs and labels
                        inputs = batch['eeg']
                        # min_vals = inputs .amin(dim=(1, 2), keepdim=True)
                        # max_vals = inputs .amax(dim=(1, 2), keepdim=True)
                        # scaling_denominator = torch.where(max_vals == min_vals, max_vals + 1e-6, max_vals - min_vals)
                        # inputs = (inputs  - min_vals) / scaling_denominator
                        # inputs = (inputs - inputs.mean(dim=(1, 2), keepdim=True)) / inputs.std(dim=(1, 2), keepdim=True)
                        # cfms = batch['cfm']
                        # mels = batch['mel_spec']
                        labels = batch['label']
                        
                        # Move to device
                        inputs = inputs.to(self.args.device)
                        
                        # cfms = cfms.to(self.args.device)
                        # mels = mels.to(self.args.device)
                        labels = labels.to(self.args.device)
                        # if inputs.size(0)!=mels.size(0):
                        #     continue
                        
                        #labels = labels.squeeze()

                        # Forward
                        # print(cfms.shape)
                        outputs = net(inputs)
                        # print(f'[OUTPUTS]{outputs.shape}')
                        # print(f'[LABELS]{labels.shape}')
                        # contrastive_loss = 0
                        # if split == 'train':
                            
                        #     hidden_dim = 256
                        #     std_range=(0.0, 1.0) 
                        #     phase_noise_magnitude_range=(0.0, 1.0)
                        #     random_state=42
                        #     bs, n_c, n_t = inputs.shape
                        #     all_views = torch.zeros(bs, 7, hidden_dim)
                        #     for i in range(7):
                        #         std = random.uniform(*std_range)
                        #         augmented_eeg = gaussian_noise(inputs,std=std,random_state=random_state)
                        #         phase_noise_magnitude = random.uniform(*phase_noise_magnitude_range)
                        #         augmented_eeg, _ = ft_surrogate(augmented_eeg,labels,phase_noise_magnitude=phase_noise_magnitude,channel_indep=False,random_state=random_state)
                        #         _, embeddings = net(augmented_eeg)
                        #         all_views[:, i, :] = embeddings
                        #     all_views = F.normalize(dim=2)
                        #     contrastive_loss = self.contrastive_criterion(all_views,labels)
                        
                        # Check NaN
                        if torch.isnan(outputs).any():
                            raise FloatingPointError('Found NaN values')
                        
                        # if torch.isnan(feats).any():
                        #     raise FloatingPointError('Found NaN values')
                        
                        # Compute loss
                        loss = self.criterion(outputs, labels)
                        # contrastive_loss = self.contrastive_criterion(feats,labels)
                        # if split == 'train':
                        #     loss = loss + contrastive_loss 
                        # needs to augment to generate `n_views` for SupCon

                        # Optimize
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        
                        # Initialize metrics
                        batch_metrics = {
                            'loss': loss.item(),
                        }
                        
                        if self.args.use_voting and split != 'train':
                            if self.args.voting_strategy == 'mean':
                                outputs = outputs.mean(dim=0)
                            elif self.args.voting_strategy == 'max':
                                outputs, _ = outputs.max(dim=0)
                            elif self.args.voting_strategy == 'min':
                                outputs, _ = outputs.min(dim=0)
                            elif self.args.voting_strategy == 'median':
                                outputs, _ = outputs.median(dim=0)
                            elif self.args.voting_strategy == 'majority':
                                try:
                                    outputs = outputs.argmax(dim=1).mode().values[0]
                                except IndexError:
                                    outputs = outputs.argmax(dim=1).mode().values
                            else:
                                raise ValueError(f"Voting strategy {self.args.voting_strategy} not recognized")
                            
                            outputs = outputs.unsqueeze(0)
                            labels = labels[0].unsqueeze(0)

                        epoch_labels.append(labels)
                        epoch_outputs.append(outputs)
                        
                        # Add metrics to epoch results
                        for k, v in batch_metrics.items():
                            v *= inputs.shape[0]
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]
          
                    # Compute Epoch metrics
                    num_samples = len(loaders[split].dataset) if not loaders[split].drop_last else len(loaders[split]) * self.args.batch_size
                    for k, v in epoch_metrics.items():
                        epoch_metrics[k] = sum(v) / num_samples
                        # Add to Saver
                        self.saver.add_scalar(f"{split}/{k}", epoch_metrics[k], epoch)
                    
                    # Aggregate logits and labels
                    epoch_labels = torch.cat(epoch_labels)
                    epoch_outputs = torch.cat(epoch_outputs)
                    if epoch_outputs.dim() > 1:
                        epoch_outputs = epoch_outputs.argmax(dim=1)

                    # Accuracy
                    accuracy = metrics.accuracy_score(epoch_labels.cpu(), epoch_outputs.cpu())
                    epoch_metrics['accuracy'] = accuracy
                    self.saver.add_scalar(f"{split}/accuracy", accuracy, epoch)
                    
                    # Balanced accuracy
                    balanced_accuracy = metrics.balanced_accuracy_score(epoch_labels.cpu(), epoch_outputs.cpu())
                    epoch_metrics['balanced_accuracy'] = balanced_accuracy
                    self.saver.add_scalar(f"{split}/balanced_accuracy", balanced_accuracy, epoch)

                    conf_matrix = metrics.confusion_matrix(epoch_labels.cpu(), epoch_outputs.cpu())
                    print(f"Confusion Matrix for {split} split:\n{conf_matrix}")

                    # Classification Report
                    class_report = metrics.classification_report(epoch_labels.cpu(), epoch_outputs.cpu(), zero_division=0)
                    print(f"Classification Report for {split} split:\n{class_report}")
                    
                    # Update result metrics
                    for metric in epoch_metrics:
                        if metric not in result_metrics[split]:
                            result_metrics[split][metric] = [epoch_metrics[metric]]
                        else:
                            result_metrics[split][metric].append(epoch_metrics[metric])

                    # Plot confusion matrix
                    #self.saver.add_confusion_matrix(
                    #    f"{split}/confusion_matrix", 
                    #    epoch_labels.cpu().tolist(), 
                    #    epoch_outputs.cpu().tolist(), 
                    #    epoch
                    #)

                # Add learning rate to saver
                self.saver.add_scalar("lr", optim.param_groups[0]['lr'], epoch)

                # Update best metrics
                
                # Lowest train loss
                if result_metrics['train']['loss'][-1] < lowest_train_loss:
                    lowest_train_loss = result_metrics['train']['loss'][-1]
                self.saver.add_scalar(f"train/lowest_loss", lowest_train_loss, epoch)
                
                # Compute validation metrics (across all validation splits)
                val_splits = [split for split in splits if 'val' in split]
                if 'val' not in result_metrics:
                    result_metrics['val'] = {
                        k: [] for k in result_metrics[val_splits[0]]
                    }
                for k in result_metrics[val_splits[0]]:
                    result_metrics['val'][k].append(sum(result_metrics[split][k][-1] for split in val_splits) / len(val_splits))
                    self.saver.add_scalar(f"val/{k}", result_metrics['val'][k][-1], epoch)

                # Max Validation accuracy
                if 'val' in result_metrics and result_metrics['val']['accuracy'][-1] > max_val_accuracy:
                    max_val_accuracy = result_metrics['val']['accuracy'][-1]
                self.saver.add_scalar(f"val/max_accuracy", max_val_accuracy, epoch)

                # Max Validation balanced accuracy
                if 'val' in result_metrics and result_metrics['val']['balanced_accuracy'][-1] > max_val_accuracy_balanced:
                    max_val_accuracy_balanced = result_metrics['val']['balanced_accuracy'][-1]
                    #test_accuracy_balanced_at_max_val_accuracy_balanced = result_metrics['test']['balanced_accuracy'][-1]
                    # Save model
                    # save the best model
                    self.saver.save_model(net, self.args.model, epoch, model_name=f"{self.args.model}")
                #self.saver.add_scalar(f"test/acc_balanced_at_max_val_acc_balanced", test_accuracy_balanced_at_max_val_accuracy_balanced, epoch)
                self.saver.add_scalar(f"val/max_balanced_accuracy", max_val_accuracy_balanced, epoch)

                # log all metrics
                self.saver.log()            

                # Check LR scheduler
                if scheduler is not None:
                    scheduler.step()
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass

        except FloatingPointError as err:
            print(f'Error: {err}')
        
        # Print main metrics
        print(f'Max val. accuracy:      {max_val_accuracy:.4f}')
        print(f'Max val. balanced acc.: {max_val_accuracy_balanced:.4f}')
        
        return net, result_metrics
    
    def test(self, test_loaders):
        
        # Setup model
        module = getattr(models, self.args.model)
        net = getattr(module, "Model")(vars(self.args))
        
        # Check resume
        if self.args.resume is not None:
            checkpoint = os.path.join(self.args.resume, f"{self.args.model}.pth")
            state_dict = torch.load(checkpoint)
            net.load_state_dict(state_dict)

        # Move to device
        net.to(self.args.device)
        
        # Set network mode
        net.eval()
        torch.set_grad_enabled(False)
        
        predictions = {}
        
        # Process each batch
        for split, test_loader in test_loaders.items():
            
            # Initialize predictions
            split_predictions = []
            
            # Process each batch
            for batch in tqdm(test_loader):
                
                # batch = self.windowing(batch) no windowing for cfm
                
                # Get inputs and labels
                inputs = batch['eeg']
                # min_vals = inputs .amin(dim=(1, 2), keepdim=True)
                # max_vals = inputs .amax(dim=(1, 2), keepdim=True)
                # scaling_denominator = torch.where(max_vals == min_vals, max_vals + 1e-6, max_vals - min_vals)
                # inputs = (inputs  - min_vals) / scaling_denominator
                # inputs = (inputs - inputs.mean(dim=(1, 2), keepdim=True)) / inputs.std(dim=(1, 2), keepdim=True)
                # cfms = batch['cfm']
                # mels = batch['mel_spec']
                
                # Move to device
                inputs = inputs.to(self.args.device)
                # cfms = cfms.to(self.args.device)
                # mels = mels.to(self.args.device)
                
                # Forward
                outputs = net(inputs)
                
                # Check NaN
                if torch.isnan(outputs).any():
                    raise FloatingPointError('Found NaN values')
                
                # Predictions
                if self.args.voting_strategy == 'mean':
                    prediction = outputs.mean(dim=0).argmax()
                elif self.args.voting_strategy == 'max':
                    prediction, _ = outputs.max(dim=0)
                    prediction = prediction.argmax()
                elif self.args.voting_strategy == 'min':
                    prediction, _ = outputs.min(dim=0)
                    prediction = prediction.argmax()
                elif self.args.voting_strategy == 'median':
                    prediction, _ = outputs.median(dim=0)
                    prediction = prediction.argmax()
                elif self.args.voting_strategy == 'majority':
                    try:
                        prediction = outputs.argmax(dim=1).mode().values[0]
                    except IndexError:
                        prediction = outputs.argmax(dim=1).mode().values
                else:
                    raise ValueError(f"Voting strategy {self.args.voting_strategy} not recognized")

                split_predictions.append(prediction.item())
                
            predictions[split] = split_predictions
            
        return predictions

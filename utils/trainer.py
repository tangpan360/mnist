import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.cuda import amp
from utils import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Trainer(object):

    def __init__(self, options: dict, model: nn.Module) -> None:
        super(Trainer, self).__init__()

        self.device = options['device']

        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'])
        self.loss_calculator = nn.CrossEntropyLoss().to(self.device)
        self.amp = options['auto_mixed_precision']
        self.scaler = amp.GradScaler(enabled=self.amp)
        self.if_step_lr = options['if_step_lr']
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                   step_size=options['lr_change_step'],
                                                   gamma=options['lr_change_gamma'])
        # print('{}\n'.format(self.scheduler.get_last_lr()[0]))

        self.loss_save_path = options['loss_path']
        self.model_save_path = options['model_path']

        self.early_stop = EarlyStopping(patience=options['patience'])

        self.train_class_loss_list = list()
        self.eval_class_loss_list = list()
        self.train_accuracy_list = list()
        self.eval_accuracy_list = list()

    def _train_one_epoch(self, train_loader, eval_loader, test_loader):

        self.train_class_loss = 0.
        self.train_class_accuracy = 0.
        self.eval_class_loss = 0.
        self.eval_class_accuracy = 0.

        self.model.train()

        for batch in train_loader:
            train_inputs, train_labels = batch

            self.optimizer.zero_grad()

            correct = 0
            total = 0

            with amp.autocast(enabled=self.amp):
                class_outputs = self.model(train_inputs.to(self.device))
                _, label_predict = torch.max(class_outputs, 1)

                loss = self.loss_calculator(class_outputs, train_labels.to(self.device))

                total += train_labels.size(0)
                correct += (label_predict == train_labels.to(self.device)).sum().item()

                self.train_class_loss += loss.item() / len(train_loader)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.train_class_accuracy = correct / total
        self.train_accuracy_list.append(self.train_class_accuracy)
        self.train_class_loss_list.append(self.train_class_loss)

        if self.if_step_lr:
            self.scheduler.step()

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                eval_inputs, eval_labels = batch

                class_outputs = self.model(eval_inputs.to(self.device))
                _, label_predict = torch.max(class_outputs, 1)

                total += eval_labels.size(0)
                correct += (label_predict == eval_labels.to(self.device)).sum().item()

                loss = self.loss_calculator(class_outputs, eval_labels.to(self.device))

                self.eval_class_loss += loss.item() / len(eval_loader)

            self.eval_class_accuracy = correct / total
            self.eval_accuracy_list.append(self.eval_class_accuracy)
            self.eval_class_loss_list.append(self.eval_class_loss)

        # self.model.eval()
        #
        # class_predict_list = list()
        # class_gt_list = list()
        #
        # with torch.no_grad:
        #     for batch in test_loader:
        #         test_inputs = batch[0]
        #         test_labels = batch[1]
        #
        #         class_outputs = self.model(test_inputs.to(self.device))
        #
        #         _, label_predict = torch.max(class_outputs, 1)
        #
        #         class_predict_list.append(label_predict)

        return self.eval_class_loss, self.eval_class_accuracy

    def train(self, options: dict, train_loader, eval_loader, test_loader):

        epochs = options['max_epoch']

        with tqdm(total=epochs, desc='Model training') as pbar:
            for epoch in range(1, epochs+1):
                eval_loss, accuracy = self._train_one_epoch(train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader)

                self.early_stop(val_loss=eval_loss, model=self.model,
                                path=os.path.join(self.model_save_path, f'{self.model.name()}_{epoch}.pt'),
                                filepath=self.model_save_path)

                tqdm.write(f'\nepoch: {epoch:5} | lr: {self.scheduler.get_last_lr()[0]/0.95:.16f} | '
                           f'train loss: {self.train_class_loss:.16f} | eval loss: {self.eval_class_loss:.16f} | '
                           f'train accuracy: {self.train_class_accuracy:.4f} | '
                           f'eval accuracy: {self.eval_class_accuracy:.4f}')

                pbar.update(1)

                if self.early_stop.early_stop:
                    print('\nEarly stopping')

                    break

        loss_dict = {
            'train_class_loss_array': np.array(self.train_class_loss_list),
            'eval_class_loss_array': np.array(self.eval_class_loss_list),
            'train_accuracy': np.array(self.train_accuracy_list),
            'eval_accuracy': np.array(self.eval_accuracy_list),
        }

        joblib.dump(loss_dict, os.path.join(self.loss_save_path, 'loss.dict'))

    def test(self, weight_file_path: str, test_loader):

        for name in os.listdir(weight_file_path):
            if name[-2:] == 'pt':
                file_path = os.path.join(weight_file_path, name)

        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                test_inputs, test_labels = batch

                class_outputs = self.model(test_inputs.to(self.device))
                _, label_predict = torch.max(class_outputs, 1)

                total += test_labels.size(0)
                correct += (label_predict == test_labels).sum().item()

        test_class_accuracy = correct / total
        print(f'Class classification accuracy in test dataset: {test_class_accuracy}')

import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True,amp = False,ocr = False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.ocr = ocr
        if eval(torch.__version__[:3])<1.6:
            self.amp = False
            print('torch version is too lower.If you want us amp Please updata torch to 1.6.0')
        else :
            if amp :
                self.amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else :
                self.amp = False
                self.scaler = None


    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        if self.amp:
            with torch.cuda.amp.autocast():
                prediction = self.model.forward(x)
                if self.ocr:
                   loss1 = self.loss(prediction[0],y)
                   loss2 = self.loss(prediction[1],y)
                   loss = loss1+loss2
                   prediction = prediction[1].float()
                else :
                    loss = self.loss(prediction, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return loss,prediction.float()
        else:
            prediction = self.model.forward(x)
            if self.ocr:
                loss1 = self.loss(prediction[0], y)
                loss2 = self.loss(prediction[1], y)
                prediction = prediction[1]
                loss = loss1 + loss2
            else:
                loss = self.loss(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss,prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True,ocr = False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.ocr = ocr

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        if self.ocr:
            loss1 = self.loss(prediction[0], y)
            loss2 = self.loss(prediction[1], y)
            loss = loss1+loss2
            return loss,prediction[1]
        else :
            self.loss(prediction, y)
            return loss, prediction

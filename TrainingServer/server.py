import grpc
import training_pb2_grpc
from ultralytics import YOLO
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from ultralytics.engine.trainer import BaseTrainer
from training_pb2 import InTrainingResponse, Metric


class TrainingServicer(training_pb2_grpc.TrainingServiceServicer):
    progress_queue = Queue()

    def __init__(self):
        super().__init__()
        self.model = YOLO("yolov8m.pt")
        self.model.add_callback("on_train_epoch_end", TrainingServicer.on_train_epoch_end)
        self.model.add_callback("on_train_end", TrainingServicer.on_train_end)

    def run_training(self):
        metrics = self.model.train(data="data.yaml", epochs=15, device=0)

    @staticmethod
    def on_train_epoch_end(trainer: BaseTrainer):
        print("Putting values into queue")
        TrainingServicer.progress_queue.put((trainer.epoch, trainer.metrics))

    @staticmethod
    def on_train_end(trainer: BaseTrainer):
        print("Training Finished")
        TrainingServicer.progress_queue.put(None)  # finished training, break infinite loop

    def StartTraining(self, request, context):
        training_thread = Process(target=self.run_training)
        training_thread.start()

        while True:
            try:
                # progress_queue never receives any items
                item = TrainingServicer.progress_queue.get(timeout=1)
                if item is None:
                    break
                epoch, metrics = item
                resp = InTrainingResponse(epoch=epoch)
                for k, v in metrics.items():
                    resp.metrics.append(Metric(name=k, value=v))
                print("Yielding training update")
                yield resp
            except Exception as e:
                print("Queue is empty or no new data available")
                continue


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    training_pb2_grpc.add_TrainingServiceServicer_to_server(TrainingServicer(), server)
    server.add_secure_port('[::]:50051', grpc.local_server_credentials())
    print("Started Server")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

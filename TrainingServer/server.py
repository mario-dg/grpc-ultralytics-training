import grpc
import training_pb2_grpc
from ultralytics import YOLO
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from training_pb2 import InTrainingResponse, Metric

from custom_trainer import CustomDetectionTrainer


class TrainingServicer(training_pb2_grpc.TrainingServiceServicer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def run_training(progress_queue: Queue):
        model = YOLO("yolov8m.pt")
        model.add_callback("on_train_epoch_end", TrainingServicer.on_train_epoch_end)
        model.add_callback("on_train_end", TrainingServicer.on_train_end)
        model.train(CustomDetectionTrainer, data="data.yaml", epochs=15, device=0, progress_queue=progress_queue)

    @staticmethod
    def on_train_epoch_end(trainer: CustomDetectionTrainer):
        print("Putting values into queue")
        trainer.progress_queue.put((trainer.epoch, trainer.metrics))

    @staticmethod
    def on_train_end(trainer: CustomDetectionTrainer):
        print("Training Finished")
        trainer.progress_queue.put(None)  # finished training, break infinite loop

    def StartTraining(self, request, context):
        progress_queue = Queue()

        training_thread = Process(target=self.run_training, args=(progress_queue,))
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

        training_thread.join()


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    training_pb2_grpc.add_TrainingServiceServicer_to_server(TrainingServicer(), server)
    server.add_secure_port('[::]:50051', grpc.local_server_credentials())
    print("Started Server")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

using Grpc.Net.Client;
using Grpc.Core;
using TrainingClient;

using var channel = GrpcChannel.ForAddress("http://localhost:50051");
var client = new TrainingService.TrainingServiceClient(channel);

using var call = client.StartTraining(new StartTrainingRequest());
await foreach (var epoch in call.ResponseStream.ReadAllAsync())
{
	Console.WriteLine($"Received Epoch {epoch.Epoch} with {epoch.Metrics}");
}

Console.WriteLine("Finished Training");
Console.ReadKey();
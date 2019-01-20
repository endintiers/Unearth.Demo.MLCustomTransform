using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;
using Unearth.Demo.MLCustomTransform.CustomTransform;
using Unearth.Demo.MLCustomTransform.Models;

namespace Unearth.Demo.MLCustomTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            var txtFeatModel = TrainModel();
            var txtFeatAccuracy = TestModel(txtFeatModel);
            Console.WriteLine("Finished");
            Console.ReadLine();

        }

        private static ITransformer TrainModel(bool useCharGrams = false)
        {
            var mlContext = new MLContext(seed: 0);

            // Create a view of the training data
            var dataPath = @"TrainingData\FlightCodes.csv";
            TextLoader textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column("FlightCode", DataKind.Text, 0),
                        new TextLoader.Column("IATACode", DataKind.Text, 1),
                    }
            });
            var trainingDataView = textLoader.Read(dataPath);

            EstimatorChain<ITransformer> dataProcessPipeline = null;

            dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("IATACode", "Label")
                .Append(mlContext.Transforms.CustomMapping<FlightCodeCMInput, FlightCodeCMOutput>(FlightCodeMapping.Transform, nameof(FlightCodeMapping)))
                .Append(mlContext.Transforms.Text.FeaturizeText("FlightCode", "FlightCodeFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "FlightCodeFeaturized", "SpecialFeature"));

            Console.WriteLine("Training the Model");

            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features);

            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Do the actual training, reads the features and builds the model
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Training took {elapsedMs / 1000f} secs");
            Console.WriteLine();

            return trainedModel;
        }

        private static float TestModel(ITransformer model)
        {
            // Create an ML.NET environment
            var mlContext = new MLContext(seed: 0);

            // Make a predictor using the trained model
            var flightCodePredictor = model.CreatePredictionEngine<FlightCodeFeatures, FlightCodePrediction>(mlContext);

            // Test the predictor (on data not used for training)
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Predicting IATA Aircraft Codes");
            Console.ForegroundColor = defaultColor;

            var correct = 0;
            var incorrect = 0;

            using (TextReader reader = new StreamReader(@"TrainingData\MoreFlightCodes.csv"))
            {
                var csvReader = new CsvReader(reader);
                var records = csvReader.GetRecords<FlightCodeFeatures>();
                foreach (var rec in records)
                {
                    var prediction = flightCodePredictor.Predict(rec);
                    if (prediction.IATACode == rec.IATACode)
                    {
                        correct++;
                        if (correct % 300 == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.WriteLine($"FlightCode: {rec.FlightCode}, Aircraft Code: {rec.IATACode} - Predicted Aircraft Code: {prediction.IATACode}, Confidence: {prediction.Confidence}");
                        }
                    }
                    else
                    {
                        incorrect++;
                        if (incorrect % 30 == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"FlightCode: {rec.FlightCode}, Aircraft Code: {rec.IATACode} - Predicted Aircraft Code: {prediction.IATACode}, Confidence: {prediction.Confidence}");
                        }
                    }
                }
            }
            var accuracy = (float)correct / (correct + incorrect);
            Console.ForegroundColor = defaultColor;
            Console.WriteLine($"Accuracy: {accuracy}");
            Console.WriteLine();
            return accuracy;
        }
    }
}

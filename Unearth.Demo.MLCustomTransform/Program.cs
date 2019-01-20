using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.ComponentModel.Composition.Hosting;
using System.IO;
using System.Reflection;
using Unearth.Demo.MLCustomTransform.CustomTransform;
using Unearth.Demo.MLCustomTransform.Models;

namespace Unearth.Demo.MLCustomTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            var modelFilePath = Path.Combine(GetAssemblyPath(), @"savedmodel.zip");

            // Train and save the model
            var model = TrainModel(modelFilePath);

            // Load and Test the model
            var accuracy = TestModel(modelFilePath: modelFilePath);

            Console.WriteLine("Finished");
            Console.ReadLine();

        }

        private static ITransformer TrainModel(string modelFilePath = null)
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

            if(!string.IsNullOrWhiteSpace(modelFilePath))
            {
                using (var fs = File.Create(modelFilePath))
                    mlContext.Model.Save(trainedModel, fs);
            }

            return trainedModel;
        }

        private static float TestModel(ITransformer model = null, string modelFilePath = null)
        {
            // Create an ML.NET environment
            var mlContext = new MLContext(seed: 0);

            if (!string.IsNullOrWhiteSpace(modelFilePath))
            {
                // Create a custom composition container for all our custom mapping actions.
                mlContext.CompositionContainer = new CompositionContainer(new TypeCatalog(typeof(CustomMappings)));

                // Load the model.
                using (var fs = File.OpenRead(modelFilePath))
                    model = mlContext.Model.Load(fs);
            }

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

        private static string GetAssemblyPath()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string result = Uri.UnescapeDataString(uri.Path);
            result = Path.GetDirectoryName(result);
            return result;
        }
    }
}

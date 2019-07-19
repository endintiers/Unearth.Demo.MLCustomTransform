using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Reflection;
using Unearth.Demo.MLCustomTransform.Models;
using Unearth.Demo.MLCustomTransform.CustomTransforms;

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
            TextLoader textLoader = mlContext.Data.CreateTextLoader(
                separatorChar: ',',
                hasHeader: true,
                columns: new[]
                    {
                        new TextLoader.Column("FlightCode", DataKind.String, 0),
                        new TextLoader.Column("IATACode", DataKind.String, 1),
                    }
            );
            var trainingDataView = textLoader.Load(dataPath);

            // Set the key column (IATACode), featurize the text FlightCode column (to a long) and add it to the features collection
            var dataPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(FlightCodeFeatures.IATACode))
                .Append(mlContext.Transforms.CustomMapping<FlightCodeCMInput, FlightCodeCMOutput>(FlightCodeMapping.Transform, nameof(FlightCodeMapping)))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "FlightCodeFeaturized", inputColumnName: nameof(FlightCodeFeatures.FlightCode)))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "FlightCodeFeaturized", "SpecialFeature"));

            // Optionally cache the input (used if multiple passes required)
            dataPipeline.AppendCacheCheckpoint(mlContext);

            // Define the trainer to be used
            IEstimator<ITransformer> trainer = null;
            trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();

            // Create a training pipeline that adds the trainer to the data pipeline and maps prediction to a string in the output (default name)
            var trainingPipeline = dataPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Do the actual training, reads the features and builds the model
            Console.WriteLine("Training the Model");
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Training took {elapsedMs / 1000f} secs");
            Console.WriteLine();

            if(!string.IsNullOrWhiteSpace(modelFilePath))
            {
                using (var fs = File.Create(modelFilePath))
                    mlContext.Model.Save(trainedModel, null, fs);
            }

            return trainedModel;
        }

        private static float TestModel(ITransformer model = null, string modelFilePath = null)
        {
            // Create an ML.NET environment
            var mlContext = new MLContext(seed: 0);

            if (!string.IsNullOrWhiteSpace(modelFilePath))
            {
                // Allow the new context to find the custom mapper we used in the pipeline
                mlContext.ComponentCatalog.RegisterAssembly(typeof(FlightCodeMapping).Assembly);

                // Load the model.
                DataViewSchema schema = null;
                using (var fs = File.OpenRead(modelFilePath))
                    model = mlContext.Model.Load(fs, out schema);
            }

            // Make a predictor using the trained model
            var flightCodePredictor = mlContext.Model.CreatePredictionEngine<FlightCodeFeatures, FlightCodePrediction>(model);

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

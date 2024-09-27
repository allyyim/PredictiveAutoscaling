using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

using Microsoft.Cloud.Metrics.Client;
using Microsoft.Cloud.Metrics.Client.Metrics;
using Microsoft.Cloud.Metrics.Client.Query;

using Microsoft.Online.Metrics.Serialization.Configuration;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using OxyPlot.Axes;
using LibNoise.Combiner;
using OxyPlot.Legends;


namespace PredictiveAutoscaling
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Create a new ML context
            var mlContext = new MLContext();

            // connect to the metrics store to get the data
            var connectionInfo = new ConnectionInfo();
            var reader = new MetricReader(connectionInfo);

            // Metric name where Account=MetricTeamInternalMetrics, Namespace=PlatformMetrics
            var memoryId = new MetricIdentifier("MetricTeamInternalMetrics", "PlatformMetrics", "\\Memory\\Available MBytes");
            var cpuId = new MetricIdentifier("MetricTeamInternalMetrics", "PlatformMetrics", "\\Processor(_Total)\\% Processor Time");

            // Dimension filters for the query using the __Role and Datacenter dimensions
            var dimensionFilters = new List<DimensionFilter>
            {
                DimensionFilter.CreateIncludeFilter("__Role", "Hinting", "MetricsStore"),
                DimensionFilter.CreateIncludeFilter("Datacenter")
            };

            // Get SUM and COUNT data for last 10 minutes for Top 10 time series keys for Memory
            IQueryResultListV3 memoryResults = await reader.GetTimeSeriesAsync(
                memoryId,
                dimensionFilters,
                DateTime.UtcNow.AddDays(-10),
                DateTime.UtcNow,
                new[] { SamplingType.Sum, SamplingType.Count },
                new SelectionClauseV3(new PropertyDefinition(PropertyAggregationType.Average, SamplingType.Sum), 10, OrderBy.Descending)
            );

            // Get SUM and COUNT data for last 10 minutes for Top 10 time series keys for CPU
            IQueryResultListV3 cpuResults = await reader.GetTimeSeriesAsync(
                cpuId,
                dimensionFilters,
                DateTime.UtcNow.AddDays(-10),
                DateTime.UtcNow,
                new[] { SamplingType.Sum, SamplingType.Count },
                new SelectionClauseV3(new PropertyDefinition(PropertyAggregationType.Average, SamplingType.Sum), 10, OrderBy.Descending)
            );


            // Extract data for ML.NET
            var observabilityData = memoryResults.Results.Zip(cpuResults.Results, (memorySeries, cpuSeries) => new ObservabilityData
            {
                MemoryUtilization = (float)memorySeries.GetTimeSeriesValues(SamplingType.Sum).Average(),
                CpuUsage = (float)cpuSeries.GetTimeSeriesValues(SamplingType.Sum).Average()
            }).ToList();

            // Load data into IDataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable(observabilityData);

            var options = new LbfgsPoissonRegressionTrainer.Options
            {
                OptimizationTolerance = 1e-4f,
                L1Regularization = 0.1f,
                L2Regularization = 0.1f,
                HistorySize = 15
            };

            var pipeline = mlContext.Transforms.Concatenate("Features", "MemoryUtilization", "CpuUsage")
            //add normalization and standardization to the pipeline
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(options));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Use the model for predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ObservabilityData, AutoscalingPrediction>(model);

            // Predict autoscaling factor based on queried data
            var prediction = predictionEngine.Predict(new ObservabilityData { MemoryUtilization = 0.5f, CpuUsage = 0.5f });
            //the predicted autoscaling factor based on the top 10 time series keys for memory and cpu write it all to a csv file
            Console.WriteLine($"Predicted autoscaling factor for the top 10 time series keys for memory and cpu: {prediction.AutoscalingFactor}");

            //write how the time series key determines the autoscaling factor in the prediction
            using (var writer = new StreamWriter("prediction.csv"))
            {
                writer.WriteLine("MemoryUtilization,CpuUsage,AutoscalingFactor");
                for (int i = 0; i < memoryResults.Results.Count; i++)
                {
                    var memorySeries = memoryResults.Results[i];
                    var cpuSeries = cpuResults.Results[i];
                    var memoryUtilization = (float)memorySeries.GetTimeSeriesValues(SamplingType.Sum).Average();
                    var cpuUsage = (float)cpuSeries.GetTimeSeriesValues(SamplingType.Sum).Average();
                    var autoscalingFactor = predictionEngine.Predict(new ObservabilityData { MemoryUtilization = memoryUtilization, CpuUsage = cpuUsage }).AutoscalingFactor;
                    writer.WriteLine($"{memoryUtilization},{cpuUsage},{autoscalingFactor}");
                }
                //plot a graph comparing the memory utilization and cpu usage and the autoscaling factor
                var resultPlotModel = new PlotModel { Title = "Memory Utilization vs CPU Usage vs Autoscaling Factor" };
                //compare the two series’ values in StairStepSeries
                var stairSeries = new StairStepSeries
                {
                    Title = "Memory Utilization vs CPU Usage",
                    MarkerType = MarkerType.Circle,
                    MarkerSize = 3,
                    MarkerStroke = OxyColors.White,
                    MarkerFill = OxyColors.Blue,
                    DataFieldX = "MemoryUtilization",
                    DataFieldY = "CpuUsage",
                    ItemsSource = observabilityData
                };
                //write a line series of how the autoscaling factor changes with the memory utilization and cpu usage
                var lineSeries = new LineSeries
                {
                    Title = "Autoscaling Factor over Time",
                    Color = OxyColors.Red,
                    MarkerType = MarkerType.Circle,
                    MarkerSize = 3,
                    MarkerStroke = OxyColors.White,
                    MarkerFill = OxyColors.Red,
                    DataFieldX = "MemoryUtilization",
                    DataFieldY = "CpuUsage",
                    ItemsSource = observabilityData
                };
                //read the csv file
                //add a legend to the graph for the line series and scatter series
                resultPlotModel.Legends.Add(new Legend { LegendPosition = LegendPosition.TopRight, LegendPlacement = LegendPlacement.Outside, LegendOrientation = LegendOrientation.Vertical, LegendBackground = OxyColor.FromAColor(200, OxyColors.White) });

                //add the x and y axis
                //add the line series to the plot model
                resultPlotModel.Series.Add(lineSeries);
                //add the series to the plot model
                resultPlotModel.Series.Add(stairSeries);
                //add the x and y axis
                resultPlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = "Memory Utilization" });
                resultPlotModel.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = "CPU Usage" });

                //white background
                resultPlotModel.Background = OxyColors.White;

                //  save it as an image
                var pngExporter = new PngExporter { Width = 600, Height = 400, Resolution = 96 };
                pngExporter.ExportToBitmap(resultPlotModel).Save("plot.png");
            }

            //save the model to a file in a folder called 'model'
            mlContext.Model.Save(model, dataView.Schema, "model.zip");

            //print out the model where it is saved get relative path
            Console.WriteLine("The training model is saved to {0}", Path.GetFullPath("model.zip"));
        }
    }

    public class ObservabilityData
    {
        public float Label { get; set; }
        public float MemoryUtilization { get; set; }
        public float CpuUsage { get; set; }
    }

    public class AutoscalingPrediction
    {
        [ColumnName("Score")]
        public float AutoscalingFactor { get; set; }
    }
}
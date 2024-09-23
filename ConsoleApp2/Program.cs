using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

using Microsoft.Cloud.Metrics.Client;
using Microsoft.Cloud.Metrics.Client.Metrics;
using Microsoft.Cloud.Metrics.Client.Query;

using Microsoft.Online.Metrics.Serialization.Configuration;

namespace PredictiveAutoscaling
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Create a new ML context
            var mlContext = new MLContext();

            // Query metric timeseries data
            var connectionInfo = new ConnectionInfo();
            var reader = new MetricReader(connectionInfo);

            // Metric identifiers for Memory and CPU usage
            var memoryId = new MetricIdentifier("MetricTeamInternalMetrics", "PlatformMetrics", "\\Memory\\Available MBytes");
            var cpuId = new MetricIdentifier("MetricTeamInternalMetrics", "PlatformMetrics", "\\Processor(_Total)\\% Processor Time");

            var dimensionFilters = new List<DimensionFilter>
            {
                DimensionFilter.CreateIncludeFilter("__Role", "Hinting", "MetricsStore"),
                DimensionFilter.CreateIncludeFilter("Datacenter")
            };

            // Get SUM and COUNT data for last 10 minutes for Top 10 time series keys for Memory
            IQueryResultListV3 memoryResults = await reader.GetTimeSeriesAsync(
                memoryId,
                dimensionFilters,
                DateTime.UtcNow.AddMinutes(-10),
                DateTime.UtcNow,
                new[] { SamplingType.Sum, SamplingType.Count },
                new SelectionClauseV3(new PropertyDefinition(PropertyAggregationType.Average, SamplingType.Sum), 10, OrderBy.Descending)
            );

            // Get SUM and COUNT data for last 10 minutes for Top 10 time series keys for CPU
            IQueryResultListV3 cpuResults = await reader.GetTimeSeriesAsync(
                cpuId,
                dimensionFilters,
                DateTime.UtcNow.AddMinutes(-10),
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
                L2Regularization = 0.1f,
                HistorySize = 15
            };

            var pipeline = mlContext.Transforms.Concatenate("Features", "MemoryUtilization", "CpuUsage")
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(options));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Use the model for predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ObservabilityData, AutoscalingPrediction>(model);

            // Predict autoscaling factor based on queried data
            var prediction = predictionEngine.Predict(new ObservabilityData { MemoryUtilization = 0.5f, CpuUsage = 0.5f });
            Console.WriteLine($"Predicted autoscaling factor: {prediction.AutoscalingFactor}");
            //save the model to a file in a folder called 'model'
            mlContext.Model.Save(model, dataView.Schema, "model.zip");
            //print out the model where it is saved get relative path
            Console.WriteLine("The training model is saved to {0}", Path.GetFullPath("model.zip"));

            //emit the autoscaling metric to Geneva Metrics account
            var autoscalingMetric = new MetricIdentifier("MetricTeamInternalMetrics", "PlatformMetrics", "\\Autoscaling\\Predicted Autoscaling Factor");
            // connect to the Geneva Metrics account




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
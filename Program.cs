using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Microsoft.ML.Data;
using Microsoft.Extensions.ML;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using System.Text.Json;
using System.Threading.Tasks;

namespace MLNETHttpApi
{
    class Program
    {
        static void Main(string[] args)
        {
            WebHost.CreateDefaultBuilder()
                .ConfigureServices(services => {
                    // Register PredictionEnginePool service 
                    services.AddPredictionEnginePool<Input,Output>()
                        .FromUri("https://github.com/dotnet/samples/raw/master/machine-learning/models/sentimentanalysis/sentiment_model.zip");
                })
                .Configure(app => {
                        app.UseHttpsRedirection();
                        app.UseRouting();
                        app.UseEndpoints(routes => {
                            // Define prediction endpoint
                            routes.MapPost("/predict", PredictHandler);
                        });                        
                })
                .Build()
                .Run();
        }

        static async Task PredictHandler(HttpContext http)
        {
            // Get PredictionEnginePool service
            var predEngine = http.RequestServices.GetRequiredService<PredictionEnginePool<Input,Output>>();

            // Deserialize HTTP request JSON body
            var input = await JsonSerializer.DeserializeAsync<Input>(http.Request.Body);

            // Predict using PredictionEnginePool service
            var prediction = predEngine.Predict(input);

            // Return prediction as response
            await http.Response.WriteAsJsonAsync(prediction);
        }
    }

    public class Input
    {
        public string SentimentText;

        [ColumnName("Label")]
        public bool Sentiment;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }    
}

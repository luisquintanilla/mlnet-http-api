using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Microsoft.ML.Data;
using Microsoft.Extensions.ML;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.OpenApi.Models;
using Ardalis.ApiEndpoints;
using Microsoft.AspNetCore.Mvc;
using Swashbuckle.AspNetCore.Annotations;

namespace MLNETHttpApi
{
    class Program
    {
        static void Main(string[] args)
        {
            WebHost.CreateDefaultBuilder()
                .ConfigureServices(services => {
                    // Add controllers
                    services.AddControllers();

                    // Add Swagger
                    services.AddSwaggerGen(c => {
                        c.SwaggerDoc("v1", new OpenApiInfo { Title = "My Sentiment Analysis API", Version = "v1" });
                        c.EnableAnnotations();
                    });  

                    // Register PredictionEnginePool service 
                    services.AddPredictionEnginePool<Input,Output>()
                        .FromUri("https://github.com/dotnet/samples/raw/master/machine-learning/models/sentimentanalysis/sentiment_model.zip");
                })
                .Configure(app => {
                        //HTTPS Redirection
                        app.UseHttpsRedirection();

                        //Swagger config
                        app.UseSwagger();
                        app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "My Sentiment Analysis API V1"));

                        // Routing config
                        app.UseRouting();
                        app.UseEndpoints(endpoints => {
                            endpoints.MapControllers();
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
        public string SentimentText {get;set;}

        [ColumnName("Label")]
        public bool Sentiment {get;set;}
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }

    // Configure API prediction handler
    public class Predict : BaseEndpoint
        .WithRequest<Input>
        .WithResponse<Output>
    {
        private readonly PredictionEnginePool<Input,Output> _predictionEnginePool;
        
        public Predict(PredictionEnginePool<Input,Output> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost("/predict")]
        [SwaggerOperation(
            Summary = "Predicts sentiment",
            Description = "Predicts sentiment",
            OperationId = "Predict")
        ]
        public override ActionResult<Output> Handle([FromBody] Input input)
        {
            return Ok(_predictionEnginePool.Predict(input));
        }
    }
}

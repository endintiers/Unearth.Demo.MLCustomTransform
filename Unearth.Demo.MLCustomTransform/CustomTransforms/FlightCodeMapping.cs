using Microsoft.ML.Transforms;
using System;

namespace Unearth.Demo.MLCustomTransform.CustomTransforms
{

    // The input columns we want to be passed to our transformer
    public class FlightCodeCMInput
    {
        public string FlightCode { get; set; }
    }

    // The output columns we want our transformer to add to the pipeline
    // If the name is the same as an existing column then that column will be replaced
    public class FlightCodeCMOutput
    {
        public float SpecialFeature { get; set; }
    }

    [CustomMappingFactoryAttribute("FlightCodeMapping")]
    public class FlightCodeMapping : CustomMappingFactory<FlightCodeCMInput, FlightCodeCMOutput>
    {
        static FlightCodeMapping()
        {
            // Init any static data needed
        }

        public static void Transform(FlightCodeCMInput input, FlightCodeCMOutput output)
        {
            // Boeing 737-800s are special (for no very good reason)
            if (input.FlightCode.Contains("B738") || input.FlightCode.Contains("73H"))
                output.SpecialFeature = 1f;
            else
                output.SpecialFeature = 0f;
        }

        public override Action<FlightCodeCMInput, FlightCodeCMOutput> GetMapping()
        {
            return Transform;
        }

    }
}

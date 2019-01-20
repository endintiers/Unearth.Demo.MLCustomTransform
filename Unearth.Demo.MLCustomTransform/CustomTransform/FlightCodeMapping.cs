using System;
using System.Collections.Generic;
using System.Text;
using Unearth.Demo.MLCustomTransform.Models;

namespace Unearth.Demo.MLCustomTransform.CustomTransform
{
 
    public class FlightCodeCMInput
    {
        public string FlightCode { get; set; }
    }

    public class FlightCodeCMOutput
    {
        public float SpecialFeature { get; set; }
    }

    public class FlightCodeMapping
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
    }
}

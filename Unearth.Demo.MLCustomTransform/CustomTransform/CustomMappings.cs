using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Text;
using Unearth.Demo.MLCustomTransform.Models;

namespace Unearth.Demo.MLCustomTransform.CustomTransform
{
    /// <summary>
    /// This class defines Exports for all custom mappings that we will use in our model(s)
    /// There is only one at the moment: FlightCodeMapping
    /// </summary>
    public class CustomMappings
    {

        // MLContext is needed to create a new transformer. We are using 'Import' to have ML.NET populate
        // this property.
        [Import]
        public MLContext MLContext { get; set; }

        // We are exporting a custom transformer by the name 'FlightCodeMapping'.
        [Export(nameof(FlightCodeMapping))]
        public ITransformer MyCustomTransformer
            => MLContext.Transforms.CustomMappingTransformer<FlightCodeCMInput, FlightCodeCMOutput>(FlightCodeMapping.Transform, nameof(FlightCodeMapping));
    }
}

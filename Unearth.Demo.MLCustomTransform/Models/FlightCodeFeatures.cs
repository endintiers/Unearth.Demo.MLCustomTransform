using Microsoft.ML.Data;

namespace Unearth.Demo.MLCustomTransform.Models
{
    public class FlightCodeFeatures
    {

        [Column(ordinal: "0")]
        public string FlightCode { get; set; }

        [Column(ordinal: "1")]
        public string IATACode { get; set; }

    }
}

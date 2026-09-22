// Test-only boundary harness. Compiles the ORIGINAL pinned C# implementations.
using System;
using System.IO;
using System.Linq;
using System.Globalization;
using System.ComponentModel.DataAnnotations;
using System.Reflection;
using System.Text;
using SmbOffice.Core.Interfaces.TimeTracking;
using SmbOffice.Core.Infrastructure.TimeTracking;
using UniversalDtoPoc.TimeTracking;

var values = File.ReadAllLines(args[0]).Select(v => v == "\\N" ? null : v).ToArray();
var fields = File.ReadAllLines(args[1]).Skip(1).Select(l => l.Split('\t')[4]).ToArray();
var mirror = new TimeTrackingEntryDto { AdditionalMetadata = new AdditionalMetadataDto() };
var children = mirror.GetType().GetProperties().Select(p => p.GetValue(mirror)).ToArray();
var smb = new TimeTrackingEntry();
for (int i = 0; i < fields.Length; ++i) {
    var child = children.Single(c => c.GetType().GetProperty(fields[i]) != null);
    var property = child.GetType().GetProperty(fields[i]);
    object value = values[i];
    if (property.PropertyType == typeof(decimal)) value = decimal.Parse(values[i], CultureInfo.InvariantCulture);
    if (property.PropertyType == typeof(bool)) value = bool.Parse(values[i]);
    property.SetValue(child, value);
    var flat = typeof(TimeTrackingEntry).GetProperty(fields[i], BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
    if (flat != null) flat.SetValue(smb, flat.PropertyType == typeof(int) ? int.Parse(values[i], CultureInfo.InvariantCulture) : value);
    Console.WriteLine(value is decimal d ? d.ToString("G29", CultureInfo.InvariantCulture) : value is bool b ? b.ToString().ToLowerInvariant() : value ?? "\\N");
}
foreach (var child in children) Validator.ValidateObject(child, new ValidationContext(child), true);
var hasher = new TimeTrackingHasher(Encoding.UTF8.GetBytes("fixture-key"));
Console.WriteLine(hasher.PrepareHashData(smb));
Console.WriteLine(hasher.CalculateHash(smb));

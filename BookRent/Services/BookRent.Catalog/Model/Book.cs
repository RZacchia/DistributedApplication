using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Catalog.Model;
[Table("Book")]
[Index(nameof(Id))]
public class Book
{
    [Key]
    public Guid Id { get; set; }
    [MaxLength(17)]
    [Required]
    public required string Isbn { get; set; }
    [MaxLength(100)]
    public required string Title { get; set; }
    [MaxLength(100)]
    public required string Author { get; set; }
    [MaxLength(255)]
    public string? Description { get; set; }
}

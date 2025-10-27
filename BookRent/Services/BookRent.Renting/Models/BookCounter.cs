using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Renting.Models;

[Table("BookCounter")]
[Index(nameof(BookId))]
public class BookCounter
{
    [Key]
    public Guid BookId { get; set; }
    public int CurrentCount { get; set; }
    public int MaxCount { get; set; }
    public DateTime UpdatedOn { get; set; }
}
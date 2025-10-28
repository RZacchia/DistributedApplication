using System.ComponentModel.DataAnnotations;

namespace BookRent.Renting.Models;

public class RentedBook
{
    [Key]
    public required Guid OrderId { get; set; }
    [Required]
    public required Guid BookId { get; set; }
    [Required]
    public required Guid UserId { get; set; }
    [Required]
    public required DateTime RentedOn { get; set; }
    [Required]
    public required DateTime DueAt { get; set; }
    public DateTime? ReturnedOn { get; set; }
    
}
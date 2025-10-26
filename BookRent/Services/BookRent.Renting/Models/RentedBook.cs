using System.ComponentModel.DataAnnotations;

namespace BookRent.Renting.Models;

public class RentedBook
{
    [Key]
    public Guid OrderId { get; set; }
    [Required]
    public Guid BookId { get; set; }
    [Required]
    public Guid UserId { get; set; }
    [Required]
    public DateTime RentedOn { get; set; }
    [Required]
    public DateTime DueAt { get; set; }
    
}
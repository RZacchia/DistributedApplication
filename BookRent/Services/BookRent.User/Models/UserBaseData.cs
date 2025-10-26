using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BookRent.User.Models;
[Table("Users")]
public class UserBaseData
{
    [Key]   
    public Guid UserId { get; init; } = Guid.NewGuid();
    [MaxLength(50), Required]
    public required string UserName { get; set; }
    [MaxLength(50), Required]
    public required string FirstName { get; set; }
    [MaxLength(50), Required]
    public required string LastName { get; set; }
    [MaxLength(50), Required]
    public required string Email { get; set; }
    
}
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BookRent.User.Models;
[Table("Users")]
public class UserBaseData
{
    [Key]   
    public required Guid UserId { get; init; }
    [MaxLength(50), Required]
    public required string UserName { get; init; }
    [MaxLength(50), Required]
    public required string FirstName { get; init; }
    [MaxLength(50), Required]
    public required string LastName { get; init; }
    [MaxLength(50), Required]
    public required string Email { get; init; }
    
}
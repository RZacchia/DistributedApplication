using BookRent.Identity.DTO.Enums;

namespace BookRent.Identity.DTO;

public record LoginRequest(string UserName, string Password);
public record RegisterOnStoreRequest(Guid Id, string Email, string Password, Role Role);
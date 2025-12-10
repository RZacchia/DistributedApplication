namespace BookRent.Orchestrator.Api.Requests;

public class AuthentificationRequest
{
    public required string Username { get; set; }
    public required string Password { get; set; }
}